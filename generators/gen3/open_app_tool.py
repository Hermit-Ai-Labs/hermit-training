#!/usr/bin/env python3
"""open_app conversations for the next Hermit run: opening sites and dApps by name or url, with the right restraint.

  python3 open_app_tool.py --seed 3 --n 300                 -> ../data3/open_app_tool.jsonl + ../eval3/open_app_tool.jsonl

Teaches:
  * known Explore name           -> open_app {"name": "<Explore spelling>"}
  * user typed an address        -> open_app {"url": "<exactly as typed>"}   (never both)
  * neither known nor given      -> one short question asking for the web address (no call)
  * unknown name                 -> open_app {"name": ...} -> not_found -> ask for the web address
  * non-http scheme              -> open_app {"url": ...} -> bad_url -> only http and https open (javascript: refused politely)
  * hard negatives (what is X, swap, deposit, add funds, buy, "open the Earn page", "is <url> the real site")
Answers are plain f-strings over the tool result and the user text. Nothing says "connected", "logged in" or "swapped".
Reuses gen2 (Conv, sim, hermit_fmt) by import; nothing in gen2 is copied or edited. World3 below is the simulator
with open_app (and v2-shaped swap/deposit/buy/add_funds/get_balances results); the eval harness can replay with
World3.from_state(record["world"]).
"""
import argparse, json, math, os, random, re, sys
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as H          # noqa: E402
import sim                      # noqa: E402
from build_agent import Conv    # noqa: E402

SPEC = os.path.join(LAB, "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
SYSTEM_V2 = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().strip("\n")
PREPARE_TOOLS = {n for n in TOOL_SCHEMA if n.startswith("prepare_")}

# ExploreData.sites from the app (wt-near HermitUI/Data/ExploreData.swift), name -> host
APPS = {
    "Hyperliquid": "app.hyperliquid.xyz", "Aave": "app.aave.com", "Pendle": "app.pendle.finance", "Ethena": "app.ethena.fi",
    "Jupiter": "jup.ag", "Uniswap": "app.uniswap.org", "Lido": "stake.lido.fi", "Pump.fun": "pump.fun",
    "Morpho": "app.morpho.org", "Kamino": "app.kamino.finance", "Polymarket": "polymarket.com", "Legion": "legion.cc",
    "Lighter": "app.lighter.xyz", "trade.xyz": "trade.xyz", "Aster": "www.asterdex.com", "Paradex": "app.paradex.trade",
    "Extended": "app.extended.exchange", "Variational": "variational.io", "Kalshi": "kalshi.com", "Predict.fun": "predict.fun",
    "Opinion": "opinion.trade", "Limitless": "limitless.exchange", "Coinbase Sales": "www.coinbase.com",
    "MetaDAO": "metadao.fi", "Echo": "echo.xyz", "Jupiter Lend": "jup.ag/lend", "Curvance": "curvance.com",
    "Kuru": "kuru.io", "Aerodrome": "aerodrome.finance", "USD.AI": "usd.ai", "Falcon": "falcon.finance",
    "Maple": "maple.finance", "Plasma": "plasma.to", "Kinetiq": "kinetiq.xyz", "Infrared": "infrared.finance",
    "Ondo": "ondo.finance", "xStocks": "xstocks.com", "BUIDL": "securitize.io", "Virtuals": "app.virtuals.io",
    "x402": "x402.org", "Infinex": "infinex.xyz",
}
APPS_LOWER = {k.lower(): k for k in APPS}
CORE = ["Hyperliquid", "Aave", "Pendle", "Ethena", "Jupiter", "Uniswap", "Lido", "Pump.fun", "Morpho", "Kamino",
        "Polymarket", "Lighter", "trade.xyz", "Aster", "Kalshi", "Aerodrome"]
WRITTEN = {  # how people type the name besides the Explore spelling (all resolve case-insensitively to the key)
    "Hyperliquid": ["hyperliquid", "Hyperliquid", "hyperliquid", "hyper liquid"],
    "Aave": ["aave", "Aave", "AAVE", "aave"], "Pendle": ["pendle", "Pendle", "pendle"], "Ethena": ["ethena", "Ethena", "ethena"],
    "Jupiter": ["jupiter", "Jupiter", "jup", "jupiter"], "Uniswap": ["uniswap", "Uniswap", "uniswap", "uni swap"],
    "Lido": ["lido", "Lido", "lido"], "Pump.fun": ["pump.fun", "pumpfun", "pump fun", "Pump.fun"],
    "Morpho": ["morpho", "Morpho", "morpho"], "Kamino": ["kamino", "Kamino", "kamino"],
    "Polymarket": ["polymarket", "Polymarket", "poly market", "polymarket"], "Lighter": ["lighter", "Lighter", "lighter"],
    "trade.xyz": ["trade.xyz", "trade xyz", "trade.xyz"], "Aster": ["aster", "Aster", "aster"],
    "Kalshi": ["kalshi", "Kalshi", "kalshi"], "Aerodrome": ["aerodrome", "Aerodrome", "aerodrome"],
    "Jupiter Lend": ["jupiter lend", "Jupiter Lend"], "Predict.fun": ["predict.fun", "predict fun"], "USD.AI": ["usd.ai", "USD.AI"],
    "Coinbase Sales": ["coinbase sales", "Coinbase Sales"], "BUIDL": ["buidl", "BUIDL"], "xStocks": ["xstocks", "xStocks"],
    "MetaDAO": ["metadao", "MetaDAO"], "Virtuals": ["virtuals", "Virtuals"], "Paradex": ["paradex", "Paradex"],
}
UNKNOWN_NAMES = ["Raydium", "Orca", "OpenSea", "Magic Eden", "Curve", "Balancer", "1inch", "Compound", "Sushi", "PancakeSwap",
                 "Meteora", "Drift", "Jito", "Marginfi", "Dexscreener", "Zora", "Farcaster", "Euler", "Frax", "GMX", "dYdX",
                 "Velodrome", "Camelot", "Moonwell", "Synthetix", "Gearbox", "Fluid", "Etherfi", "Stargate", "Across",
                 "Rabby", "Phantom", "Blur", "Tensor", "Hyperlend", "Felix", "Ember", "Silo", "Spark", "Convex"]
assert not any(u.lower() in APPS_LOWER for u in UNKNOWN_NAMES)

BANNED = ["connected", "logged in", "signed in", "swapped", "relay", "\u00b7", "i've connected", "has been sent",
          "went through", "deposited", "i bought", "have bought", "was bought", "already happened"]
CHECK_LINE = "Check the address matches the real site before you connect."
NOTHING_LINE = "Nothing connects to your wallet until you approve it."
NONHIER = {"javascript", "data", "mailto", "tel", "file", "about", "vbscript", "blob", "view-source", "sms", "magnet"}
CHAIN_LABEL = {"base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "solana": "Solana", "bsc": "BNB Chain",
               "polygon": "Polygon"}


def approx(x):
    """Units.approx from the app: >=1000 -> 2 dp, else 3 significant digits, trailing zeros trimmed."""
    if x >= 1000:
        s = f"{x:.2f}"
    else:
        s = format(Decimal(f"{x:.3g}"), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def domain(host_or_url):
    """Registrable-ish domain: last two labels of the host (enough for the sites we use)."""
    s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", host_or_url.strip())
    host = re.split(r"[/?#]", s, 1)[0].split("@")[-1].split(":")[0].lower()
    parts = [p for p in host.split(".") if p]
    return ".".join(parts[-2:])


# =================================================================== simulator
class World3(sim.World):
    """gen2 World plus open_app and v2-shaped results (route, button_shown, Buy review...)."""

    def t_open_app(self, name=None, url=None):
        if name is not None and url is not None:
            return {"error": "Give either a name or a url, not both."}
        if url is not None:
            u = str(url).strip()
            if not u:
                return {"error": "Give a name or a url."}
            m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):(.*)$", u)
            if m and (m.group(2).startswith("//") or m.group(1).lower() in NONHIER):
                if m.group(1).lower() not in ("http", "https"):
                    return {"error": "Only http and https addresses can be opened."}
                opened = u
            else:
                opened = "https://" + u
            return {"note": "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it.",
                    "opened": opened, "status": "opened"}
        if name is None or not str(name).strip():
            return {"error": "Give a name or a url."}
        key = APPS_LOWER.get(str(name).strip().lower())
        if key is None:
            return {"note": f"No dApp called {str(name).strip()} in Explore. Ask the user for the web address.", "status": "not_found"}
        return {"note": "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it.",
                "opened": "https://" + APPS[key], "status": "opened"}

    def t_get_balances(self, chain=None):
        rows = [{"amount": f"{b['amount']:g}", "chain": "bnb" if b["chain"] == "bsc" else b["chain"], "symbol": b["symbol"],
                 "usd": b["usd"]} for b in self.balances if b["amount"] > 0 and (chain is None or b["chain"] == str(chain).lower())]
        return {"holdings": rows}

    def t_open_add_funds(self, goal):
        if goal not in ("wallet", "hyperliquid"):
            return {"error": "goal must be wallet or hyperliquid"}
        return {"goal": goal, "note": "The user taps it to open Add funds.", "screen": "add_funds", "status": "button_shown"}

    def _have(self, symbol, chain):
        return sum(b["amount"] for b in self.balances if b["symbol"].upper() == str(symbol).upper() and b["chain"] == str(chain).lower())

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        have = self._have(from_symbol, from_chain)
        if have < float(amount):
            return {"holding": f"{have:g} {from_symbol} on {CHAIN_LABEL.get(from_chain, from_chain)}", "note": "The user holds less than that. Tell them.",
                    "prepared": False, "status": "insufficient"}
        usd_in = float(amount) * self.prices[from_symbol]
        fee_bps = 35
        cost = round(usd_in * fee_bps / 10000, 2)
        out = usd_in * (1 - fee_bps / 10000) / self.prices[to_symbol]
        r = {"amount_in_usd": fixnum(round(usd_in, 2)), "amount_out": approx(out), "amount_out_usd": fixnum(round(usd_in - cost, 2)), "fee_bps": fee_bps,
             "from": f"{amount:g} {from_symbol} on {CHAIN_LABEL.get(from_chain, from_chain)}",
             "note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.",
             "policy": "allow", "route": "NEAR Intents", "seconds": self.rng.randint(8, 30), "signature_verified": True,
             "status": "prepared_for_review",
             "to": f"about {approx(out)} {to_symbol} on {CHAIN_LABEL.get(to_chain, to_chain)}", "total_cost_usd": fixnum(cost)}
        return r

    AAVE_CHAINS = ("ethereum", "base", "arbitrum")

    def t_prepare_deposit(self, symbol, amount, project="aave-v3", chain=None):
        if str(project).lower() not in ("aave-v3", "aave", "aave v3"):
            return {"note": "Only Aave v3 deposits can be prepared here. Other pools are in Earn, DeFi, with Ask.", "prepared": False,
                    "status": "unsupported"}
        cands = [c for c in self.AAVE_CHAINS if (chain is None or c == str(chain).lower()) and self._have(symbol, c) > 0]
        if not cands:
            return {"holding": "nothing on Aave chains", "note": "The user holds less than that. Tell them.", "prepared": False,
                    "status": "insufficient"}
        if len(cands) > 1:
            return {"note": "Nothing was prepared. Ask the user which network, in one short question.", "options": cands, "prepared": False,
                    "question": "Which network should the deposit use?", "status": "needs_chain"}
        c = cands[0]
        have = self._have(symbol, c)
        if have < float(amount):
            return {"holding": f"{have:g} {symbol} on {CHAIN_LABEL[c]}", "note": "The user holds less than that. Tell them.", "prepared": False,
                    "status": "insufficient"}
        return {"amount": f"{amount:g} {symbol}", "apy_pct": round(self.rng.uniform(2.2, 7.5), 2), "executed": False, "kind": "deposit",
                "note": "Waiting for the user to review the deposit and approve it with Face ID.", "pool": f"Aave v3 on {CHAIN_LABEL[c]}",
                "status": "review_opened"}

    def t_prepare_buy(self, symbol, amount_usd, pay_with=None):
        sym = str(symbol).upper()
        if sym not in self.prices:
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": [f"There's no verified {sym} token on Solana, so Hermit won't buy one."],
                    "status": "blocked"}
        have = self._have("USDC", "solana")
        if have < float(amount_usd):
            return {"holding": f"{have:g} USDC on Solana", "note": "The user holds less than that. Tell them.", "prepared": False,
                    "status": "insufficient"}
        got = float(amount_usd) / self.prices[sym]
        return {"executed": False, "kind": "buy",
                "note": "Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with Face ID or cancels.",
                "price_impact_pct": round(self.rng.uniform(0.01, 0.3), 2), "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5,
                "status": "review_opened", "symbol": sym, "you_get": f"about {approx(got)} {sym}",
                "you_pay": f"{amount_usd:g} USDC on Solana"}


# =================================================================== text helpers
def fixnum(x):
    return int(x) if float(x).is_integer() else x


VERB_TYPOS = {"open": ["opne", "opn", "oepn", "ope"], "launch": ["lauch", "lanuch", "lunch"], "take me to": ["tkae me to", "take me too", "tak me to"],
              "pull up": ["pul up", "pull upp"], "go to": ["goto", "go too"], "connect": ["conect", "connnect"], "navigate": ["navagate"],
              "please": ["pleas", "plz"], "visit": ["viist", "vist"], "wanna": ["wana"], "show me": ["shwo me"], "swap": ["swp", "sawp"],
              "deposit": ["depsoit", "deposti"], "what is": ["waht is", "what si"], "what's": ["whats", "wat's"], "explain": ["explian"]}
PRE = ["", "", "", "", "", "hey ", "yo ", "ok ", "hi, ", "pls ", "hey hermit, ", "hermit ", "ok so ", "quick one, ", "hello "]
SUF = ["", "", "", "", "", " pls", " please", " thanks", " now", " quickly", " for me", " asap"]


def typo_verbs(rng, t):
    if rng.random() < 0.14:
        for k in VERB_TYPOS:
            if re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", t):
                return re.sub(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", rng.choice(VERB_TYPOS[k]), t, 1)
    return t


def compose(rng, tpls, allow_punct=True, **kw):
    """Pick a template, style it, fill placeholders. Placeholder values (urls, names, addresses) are never altered."""
    t = typo_verbs(rng, rng.choice(tpls))
    if rng.random() < 0.4:
        t = t.lower()
    t = rng.choice(PRE) + t + rng.choice(SUF)
    if allow_punct and not t.endswith("?"):
        r = rng.random()
        t += "." if r < 0.08 else ("!" if r < 0.12 else "")
    t = t.format(**kw)
    t = re.sub(r"  +", " ", t).strip()
    return t[0].upper() + t[1:] if (t and rng.random() < 0.3 and t[0].isalpha() and not kw.get("_keepcase")) else t


def written_name(rng, name):
    """(text as typed, was_typo). Mild typo only for long plain names."""
    forms = WRITTEN.get(name) or [name.lower(), name]
    w = rng.choice(forms)
    if len(w) >= 7 and w.isalpha() and rng.random() < 0.08:
        i = rng.randint(2, len(w) - 3)
        w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return w, True
    return w, False


def pick_name(rng):
    if rng.random() < 0.8:
        return rng.choice(CORE)
    return rng.choice(list(APPS))


def new_conv(rng, w=None):
    bio = "Face ID" if rng.random() < 0.7 else "Touch ID"
    w = w or World3(rng)
    c = Conv(rng, w, system=SYSTEM_V2.replace("{BIO}", bio))
    return w, c, bio


def finish(c, bio, kind, gold, **meta):
    ex = c.ex(kind)
    if bio != "Face ID":
        for m in ex["messages"]:
            if m["role"] in ("assistant", "tool"):
                m["content"] = m["content"].replace("Face ID", bio)
    ex["meta"]["source"] = "open_app_tool"
    ex["meta"]["bio"] = bio
    ex["meta"]["gold"] = gold
    ex["meta"].update(meta)
    return ex


def last_user(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "user"][-1]


def answer_gold(**kw):
    return {"must_not_include": BANNED, **kw}


# =================================================================== open_app positives
NAME_TPLS = [
    "open {n}", "Open {n}", "take me to {n}", "go to {n}", "pull up {n}", "launch {n}", "I want to use {n}", "connect my wallet to {n}",
    "open {n} pls", "can you open {n}", "could you take me to {n}", "show me the {n} site", "open the {n} app", "open {n} app",
    "bring up {n}", "load {n}", "let's go to {n}", "head to {n}", "navigate to {n}", "i wanna use {n}", "wanna check out {n}",
    "open up {n}", "fire up {n}", "jump into {n}", "hey open {n}", "yo take me to {n}", "open {n}'s website", "can you open {n} for me?",
    "open {n} in the browser", "open {n} in hermit's browser", "launch the {n} dapp", "pull up the {n} dapp", "open {n} dapp",
    "i'd like to use {n}", "connect to {n}", "link my wallet to {n}", "log me into {n}", "sign me into {n}", "{n} time, open it",
    "take me to the {n} app", "go to the {n} website", "open {n} real quick", "i need to get on {n}", "get me to {n}", "let me use {n}",
    "start {n}", "open {n} for me", "show me {n}", "browse to {n}", "take me over to {n}", "can we go on {n}", "go on {n}",
    "use {n}", "i want {n} open", "{n} dapp please", "can I get {n} up", "bring me to {n}", "pls launch {n}", "get {n} open",
    "i want to trade on {n}", "i want to go on {n} and look around", "lets use {n}", "open {n}, i'm going to check something",
]
PURPOSES = ["check my positions", "look at the markets", "see the rates", "check the yields", "have a look around", "see what's trending",
            "check the fees", "read the docs there", "see the leaderboard", "see what's new"]
TICKERS = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "SPY", "QQQ", "COIN", "MSTR", "AMD"]
CONNECT_TPLS = ["connect my wallet to {n}", "Connect my wallet to {n}", "connect wallet to {n}", "i want to connect my wallet on {n}",
                "log me into {n}", "sign me into {n}", "link my wallet to {n}", "hook my wallet up to {n}", "connect to {n} with my wallet",
                "I need to connect to {n}", "get my wallet on {n}"]
INTENT_ONLY = {"connect"}


def user_open_name(rng, name):
    n, typo = written_name(rng, name)
    r = rng.random()
    if name == "trade.xyz" and r < 0.5:
        tk = rng.choice(TICKERS)
        tpl = rng.choice(["open {n} to look at {tk}", "open {n} so i can check {tk}", "take me to {n}, i want to see {tk}",
                          "pull up {n} for {tk}", "open {n} to look at {tk}"])
        return compose(rng, [tpl], n=n, tk=tk), n, typo, "look"
    if r < 0.16:
        return compose(rng, CONNECT_TPLS, n=n), n, typo, "connect"
    if r < 0.28:
        tpl = rng.choice(["open {n} to {p}", "take me to {n} so i can {p}", "pull up {n}, i want to {p}", "go to {n} to {p}",
                          "launch {n} i need to {p}"])
        return compose(rng, [tpl], n=n, p=rng.choice(PURPOSES)), n, typo, "purpose"
    return compose(rng, NAME_TPLS, n=n), n, typo, "plain"


def open_answer(rng, label, opened, mode, warn=False):
    x = label if label else opened
    if label and rng.random() < 0.18:
        x = f"{label} ({re.sub(r'^https?://', '', opened)})"
    a = rng.choice([f"I opened {x} in Hermit's browser.", f"I opened {x} in Hermit's browser.", f"I opened {x} in Hermit's browser.",
                    f"Done, {x} is open in Hermit's browser.", f"{x} is open in Hermit's browser now.", f"Opened {x} in Hermit's browser."])
    parts = [a, NOTHING_LINE]
    if mode == "connect":
        parts.append(rng.choice(["When the site asks for your wallet, Hermit shows an approval sheet first.",
                                 "If the site asks to connect, Hermit asks you to approve it first.",
                                 "Use the site's own wallet button when you're ready, and Hermit will ask you to approve it."]))
    if warn:
        parts.append(CHECK_LINE)
    return " ".join(parts)


def sc_open_name(rng):
    w, c, bio = new_conv(rng)
    name = pick_name(rng)
    text, nm, typo, mode = user_open_name(rng, name)
    c.user(text)
    args = {"name": name}
    r = c.call(("open_app", args))
    assert r["status"] == "opened", r
    c.say(open_answer(rng, name, r["opened"], mode))
    gold = {"tool": "open_app", "name": name, "url": None, "calls": [{"name": "open_app", "arguments": args}],
            "answer": answer_gold(must_include_any=[name, r["opened"]], must_include=[NOTHING_LINE], must_not_include_extra=[CHECK_LINE])}
    return finish(c, bio, "open_name", gold)


COMMON_SITES = [
    "example.com/app", "example.com", "defillama.com", "defillama.com/yields", "dexscreener.com/solana", "coingecko.com",
    "www.coingecko.com/en/coins/bitcoin", "etherscan.io", "basescan.org", "solscan.io", "l2beat.com", "rekt.news", "hermit.my",
    "github.com/Hermit-Ai-Labs", "news.ycombinator.com", "en.wikipedia.org/wiki/Bitcoin", "across.to", "app.1inch.io", "app.ens.domains",
    "snapshot.org", "opensea.io", "magiceden.io", "raydium.io/swap", "orca.so", "curve.fi", "app.balancer.fi", "stargate.finance",
    "revoke.cash", "tokenterminal.com", "cryptopanic.com", "my-dao.xyz/vote", "app.compound.finance", "app.gmx.io", "dydx.trade",
    "app.meteora.ag", "drift.trade", "www.jito.network", "tensor.trade", "velodrome.finance", "app.pancakeswap.finance",
    "docs.uniswap.org", "docs.aave.com", "blog.example.org/post/12", "my-project.xyz/app?ref=abc", "status.example.com",
    "dune.com/browse/dashboards", "arbiscan.io", "polygonscan.com", "bscscan.com", "coinmarketcap.com", "tradingview.com/chart",
    "www.tradingview.com/symbols/BTCUSD", "vitalik.eth.limo", "farcaster.xyz", "warpcast.com", "jup.ag/swap/SOL-USDC", "jup.ag/lend",
    "app.uniswap.org/swap", "app.aave.com/markets", "app.hyperliquid.xyz/trade", "polymarket.com/markets", "kalshi.com/markets",
    "app.morpho.org/ethereum/earn", "app.pendle.finance/trade/markets", "stake.lido.fi/withdrawals", "app.lighter.xyz/trade/ETH",
    "aerodrome.finance/swap", "pump.fun/board", "app.kamino.finance/lend",
]
SCHEMES = ["https://", "https://", "https://", "http://", "", "", ""]


def pick_url(rng, avoid_names=True):
    r = rng.random()
    if r < 0.07:
        base = "etherscan.io/address/" + H.evm_addr(rng)
    elif r < 0.12:
        base = "solscan.io/account/" + H.sol_addr(rng)
    elif r < 0.2:
        base = "https://" + rng.choice(["www.", "", "app."]) + rng.choice(["alpha", "orbit", "nimbus", "quanta", "zephyr", "lumen", "fable", "kite"]) + str(
            rng.randint(1, 99)) + rng.choice([".xyz", ".io", ".app", ".finance", ".fi", ".so"]) + rng.choice(["", "", "/app", "/dashboard", "/?ref=hermit"])
    else:
        base = rng.choice(COMMON_SITES)
    if "://" not in base:
        base = rng.choice(SCHEMES) + base
    if avoid_names and base.lower() in APPS_LOWER:
        return pick_url(rng)
    return base


URL_TPLS = [
    "open {u}", "go to {u}", "open this: {u}", "take me to {u}", "pull up {u}", "can you open {u}", "open that site: {u}", "open the site {u}",
    "visit {u}", "browse to {u}", "load {u} in the browser", "i want to see {u}", "check this out {u}", "open this link {u}",
    "got this link {u} can you open it", "my friend sent me {u} open it", "launch {u}", "{u} open it", "take me here {u}", "open {u} in hermit",
    "open {u} for me", "pls open {u}", "navigate to {u}", "can u open {u} for me", "open up {u}", "go on {u}", "open website {u}",
    "opne {u}", "show me {u}", "i need {u} open", "open this site {u} please", "open the link {u}", "bring up {u}",
]


def sc_open_url(rng):
    w, c, bio = new_conv(rng)
    r0 = rng.random()
    name_ctx = None
    warn = False
    if r0 < 0.45:
        # a known dApp, the user typed its (real) address
        name = pick_name(rng)
        host = APPS[name]
        path = rng.choice(["", "", "", "/swap", "/markets", "/trade"]) if "/" not in host else ""
        u = rng.choice(["https://", "https://", "http://", ""]) + host + path
        if u.lower() in APPS_LOWER:
            u = "https://" + u
        n, _ = written_name(rng, name)
        if rng.random() < 0.5:
            text = compose(rng, URL_TPLS, u=u)
        else:
            text = compose(rng, ["open {n} at {u}", "take me to {n}: {u}", "{n} {u} open it", "go to {n} {u}", "open {n}, the address is {u}",
                                 "i want {n}, here's the link {u}", "open the {n} site {u}"], n=n, u=u)
        name_ctx = name if (n.lower() in text.lower()) else None
        if name_ctx and domain(u) != domain(APPS[name_ctx]):
            warn = True
    else:
        u = pick_url(rng)
        text = compose(rng, URL_TPLS, allow_punct=False, u=u)
    args = {"url": u}
    c.user(text)
    r = c.call(("open_app", args))
    assert r["status"] == "opened", r
    c.say(open_answer(rng, None, r["opened"], "plain", warn))
    gold = {"tool": "open_app", "name": None, "url": u, "calls": [{"name": "open_app", "arguments": args}],
            "answer": answer_gold(must_include_any=[r["opened"], u], must_include=[NOTHING_LINE] + ([CHECK_LINE] if warn else []),
                                  must_not_include_extra=[] if warn else [CHECK_LINE])}
    return finish(c, bio, "open_url_known_domain" if name_ctx else "open_url", gold, compare_name=name_ctx)


def lookalike(rng, name):
    host = APPS[name]
    dom = domain(host)
    base = re.sub(r"[^a-z0-9]", "", dom.split(".")[0])
    if name == "trade.xyz":
        base = "tradexyz"
    pats = [f"{base}-app.xyz", f"app-{base}.org", f"{base}claim.com", f"{base}.app", f"app.{base}-finance.com", f"{base}-airdrop.io",
            f"{base}.finance", f"claim-{base}.xyz", f"{base}app.net", f"{base}-official.com", f"my{base}.io", f"{base}s.xyz", f"{base}.cc"]
    h = rng.choice(pats)
    if domain(h) == dom:
        return lookalike(rng, name)
    return rng.choice(["https://", "https://", "http://", ""]) + h + rng.choice(["", "", "/claim", "/app", "/connect"])


def sc_open_url_lookalike(rng):
    w, c, bio = new_conv(rng)
    name = pick_name(rng)
    u = lookalike(rng, name)
    n, _ = written_name(rng, name)
    text = compose(rng, [
        "open {n} at {u}", "take me to {n}: {u}", "i got this {n} link {u} open it", "open {u} i want to use {n}", "go to {n} {u}",
        "someone sent me {u} for {n} can you open it", "{n} airdrop claim {u} open", "open the {n} site {u}", "open {n} here {u}",
        "this is the {n} link right? {u} open it", "pull up {n} using {u}", "launch {n} {u}", "my friend says {n} is at {u}, open it",
        "{n} announced a new site {u} take me there"], allow_punct=False, n=n, u=u)
    args = {"url": u}
    c.user(text)
    r = c.call(("open_app", args))
    assert r["status"] == "opened", r
    c.say(open_answer(rng, None, r["opened"], "plain", True).replace(CHECK_LINE, f"That is not the {name} address in Explore, which is {APPS[name]}. " + CHECK_LINE))
    gold = {"tool": "open_app", "name": None, "url": u, "calls": [{"name": "open_app", "arguments": args}],
            "answer": answer_gold(must_include_any=[r["opened"], u], must_include=[NOTHING_LINE, CHECK_LINE])}
    return finish(c, bio, "open_url_lookalike", gold, compare_name=name)


BAD_URLS = ["javascript:alert(1)", "javascript:alert(document.cookie)", "javascript:void(0)", "ftp://files.example.com/pub", "ftp://ftp.gnu.org/gnu",
            "file:///etc/passwd", "file:///Users/me/notes.txt", "data:text/html,hello", "mailto:team@example.com", "tel:+15551234567",
            "chrome://settings", "ssh://git@github.com/x", "ws://localhost:8080", "ipfs://bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
            "telnet://towel.blinkenlights.nl", "view-source:https://example.com", "about:blank", "vbscript:msgbox(1)", "javascript:window.close()",
            "ftp://192.168.1.20/share", "file:///C:/Users/me/wallet.txt", "sftp://backup.example.com/keys"]
BAD_TPLS = ["open {u}", "go to {u}", "can you open {u}", "load {u}", "open this link {u}", "visit {u}", "take me to {u}", "run {u} in the browser",
            "open this {u} for me", "pull up {u}", "i want to open {u}", "launch {u}", "open this site: {u}", "browse to {u}", "paste this in the browser {u}"]


def sc_bad_url(rng):
    w, c, bio = new_conv(rng)
    u = rng.choice(BAD_URLS)
    text = compose(rng, BAD_TPLS, allow_punct=False, u=u)
    args = {"url": u}
    c.user(text)
    r = c.call(("open_app", args))
    assert "error" in r, r
    if u.lower().startswith(("javascript:", "vbscript:")):
        a = rng.choice([
            "I won't open that. A javascript: address runs code instead of showing a website, and only http and https addresses open in Hermit's browser. If you meant a normal site, send me its web address.",
            "I'm not going to open that one: it runs a script rather than loading a site, and only http and https addresses open here. Send me the site's normal address and I'll open it.",
            "Sorry, I can't open that. Only http and https addresses open in Hermit's browser, and a script link like this is one I'd rather not run. Send a normal web address if you have one."])
        a = a.replace("javascript: address", "script link") if u.lower().startswith("vbscript") else a
    else:
        scheme = u.split(":", 1)[0].lower()
        a = rng.choice([
            f"I can't open that one. Only http and https addresses open in Hermit's browser, and this is a {scheme} address.",
            f"Sorry, that's a {scheme} address and only http and https open here. If the site has a normal web address, send it and I'll open it.",
            "Only http and https addresses open in Hermit's browser, so I couldn't open that. Send me a web address that starts with http or https, or just the site name."])
    c.say(a)
    gold = {"tool": "open_app", "name": None, "url": u, "calls": [{"name": "open_app", "arguments": args}],
            "answer": answer_gold(must_include=["http", "https"], must_not_include_extra=["I opened", "is open in"])}
    return finish(c, bio, "open_bad_url", gold)


def sc_not_found(rng):
    w, c, bio = new_conv(rng)
    name = rng.choice(UNKNOWN_NAMES)
    n = name if rng.random() < 0.5 else name.lower()
    text = compose(rng, NAME_TPLS, n=n)
    args = {"name": name if n == name else n}
    c.user(text)
    r = c.call(("open_app", args))
    assert r["status"] == "not_found", r
    a = rng.choice([f"{args['name']} isn't in Hermit's Explore list, so what's its web address?",
                    f"I couldn't find {args['name']} in Explore. What's the web address?",
                    f"I don't have {args['name']} in Hermit's Explore list. What's its web address?",
                    f"There's no {args['name']} in Explore, so can you send me its web address?"])
    c.say(a, ask=True)
    gold = {"tool": "open_app", "name": args["name"], "url": None, "calls": [{"name": "open_app", "arguments": args}], "ask_after": True,
            "answer": answer_gold(must_include=["?"], must_not_include_extra=["I opened", "is open in"])}
    return finish(c, bio, "open_not_found", gold)


GENERIC_OPEN = [
    "open the site", "open that site", "open the website", "open that dapp", "take me to the site", "go to the website", "pull up that app",
    "open the link", "can you open that website for me", "open the dapp my friend told me about", "i want to use that new app",
    "take me to the exchange", "open the thing i was looking at", "launch the app", "go to that site i mentioned", "open the website i sent",
    "open a site for me", "can you open a website", "open the app", "launch it", "open it", "open the new dex", "take me to the lending site",
    "pull up the site", "i want to go to a site", "open my bank", "go to the faucet", "open the airdrop site", "browse to the site",
    "open that prediction market", "open the perps site", "take me to my dex", "open the explorer", "open the staking site",
]
ASK_Q = ["What's the web address of the site you want to open?", "Sure, what's the address of the site?",
         "Which site do you mean, and what's its web address?", "What's the web address of the site?", "What's the site's web address?",
         "Which website do you mean? I need its web address to open it."]


def sc_ask(rng):
    w, c, bio = new_conv(rng)
    text = compose(rng, [rng.choice(GENERIC_OPEN)])
    c.user(text)
    c.say(rng.choice(ASK_Q), ask=True)
    gold = {"tool": None, "name": None, "url": None, "calls": [], "ask": True, "answer": answer_gold(must_include=["?"], must_not_include_extra=["I opened"])}
    return finish(c, bio, "open_ask", gold)


def sc_ask_then_give(rng):
    w, c, bio = new_conv(rng)
    c.user(compose(rng, [rng.choice(GENERIC_OPEN)]))
    c.say(rng.choice(ASK_Q), ask=True)
    if rng.random() < 0.7:
        u = pick_url(rng)
        c.user(rng.choice([u, "{u}", "it's {u}", "its {u}", "here: {u}", "the address is {u}", "{u} that one", "try {u}"]).format(u=u))
        args = {"url": u}
        r = c.call(("open_app", args))
        c.say(open_answer(rng, None, r["opened"], "plain"))
        gold = {"tool": "open_app", "name": None, "url": u, "calls": [{"name": "open_app", "arguments": args}],
                "answer": answer_gold(must_include_any=[r["opened"], u], must_include=[NOTHING_LINE])}
    else:
        name = pick_name(rng)
        n, _ = written_name(rng, name)
        c.user(rng.choice(["{n}", "it's {n}", "{n} one", "the {n} one", "i mean {n}", "{n} pls", "{n}, the app"]).format(n=n))
        args = {"name": name}
        r = c.call(("open_app", args))
        c.say(open_answer(rng, name, r["opened"], "plain"))
        gold = {"tool": "open_app", "name": name, "url": None, "calls": [{"name": "open_app", "arguments": args}],
                "answer": answer_gold(must_include_any=[name, r["opened"]], must_include=[NOTHING_LINE])}
    return finish(c, bio, "open_ask_then_give", gold)


# =================================================================== hard negatives
DESC = {
    "Hyperliquid": "a perpetual futures exchange that runs on its own blockchain. Hermit's trading uses it",
    "Aave": "a lending protocol: you deposit crypto to earn interest, or borrow against what you deposit",
    "Pendle": "a protocol that splits a yield-bearing asset in two, so you can lock in a fixed yield or trade future yield",
    "Ethena": "the issuer of USDe, a synthetic dollar, plus a staked version that earns a yield. The yield can change and carries risk",
    "Jupiter": "the main swap aggregator on Solana: it looks across Solana exchanges for the best route",
    "Uniswap": "a decentralised exchange where you swap tokens straight from your wallet, with no account",
    "Lido": "a liquid staking service: you stake ETH and get stETH back, which you can keep using elsewhere",
    "Pump.fun": "a Solana launchpad where anyone can create a memecoin. Most of those coins are very risky",
    "Morpho": "a lending protocol with vaults you can deposit into to earn interest",
    "Kamino": "a lending and yield protocol on Solana",
    "Polymarket": "a prediction market where you trade on the outcome of real-world events",
    "Lighter": "a decentralised exchange for perpetual futures",
    "trade.xyz": "a market for perps that track stocks and indexes, built on Hyperliquid and open around the clock",
    "Aster": "a decentralised exchange for perpetual futures",
    "Kalshi": "a regulated prediction market where you trade on event outcomes",
    "Aerodrome": "a decentralised exchange on Base where liquidity providers earn trading fees",
}
WHAT_TPLS = ["what is {n}", "what's {n}", "whats {n}", "explain {n}", "tell me about {n}", "what does {n} do", "how does {n} work",
             "what is {n} exactly", "can you explain what {n} is", "{n}, what is it", "i keep hearing about {n}, what is it", "ELI5 {n}",
             "what's {n} used for", "who uses {n} and why", "never heard of {n}, what is it", "what is {n} in crypto", "give me a quick rundown of {n}"]
OFFERS = ["If you want to look at it, say \"open {N}\" and I'll open it in Hermit's browser.", "I can open it in Hermit's browser if you ask.",
          "Say \"open {N}\" if you want to see it."]


def sc_neg_what_is(rng):
    w, c, bio = new_conv(rng)
    name = rng.choice(list(DESC))
    n, _ = written_name(rng, name)
    c.user(compose(rng, WHAT_TPLS, n=n))
    a = f"{name} is {DESC[name]}."
    if rng.random() < 0.6:
        a += " " + rng.choice(OFFERS).format(N=name)
    c.say(a)
    gold = {"tool": None, "name": None, "url": None, "calls": [], "text": True, "answer": answer_gold(must_include=[name], must_not_include_extra=["I opened"])}
    return finish(c, bio, "neg_what_is", gold)


def sc_neg_swap(rng):
    w, c, bio = new_conv(rng)
    fs, fc, ts, tc = rng.choice([("USDC", "base", "ETH", "base"), ("USDC", "arbitrum", "ETH", "arbitrum"), ("ETH", "base", "USDC", "base"),
                                 ("USDC", "ethereum", "ETH", "ethereum"), ("ETH", "arbitrum", "USDC", "arbitrum"), ("USDC", "base", "SOL", "solana"),
                                 ("USDT", "arbitrum", "ETH", "arbitrum"), ("USDC", "arbitrum", "ARB", "arbitrum")])
    a = rng.choice([5, 10, 20, 25, 50, 100, 150, 200, 500]) if fs in ("USDC", "USDT") else rng.choice([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1])
    # the wallet holds the source token on exactly one chain
    w.balances = [b for b in w.balances if b["symbol"] != fs]
    w.ensure_balance(fs, fc, a)
    chainless = rng.random() < 0.4
    cs = CHAIN_LABEL[fc]
    cs = rng.choice([cs, cs.lower()])
    if chainless:
        text = compose(rng, ["swap {a} {f} to {t}", "swap {a} {f} for {t}", "convert {a} {f} to {t}", "exchange {a} {f} into {t}", "turn {a} {f} into {t}",
                             "i want to swap {a} {f} to {t}", "swap {a} {f} to {t} pls"], a=a, f=fs, t=ts)
        c.user(text)
        b = c.call(("get_balances", {}))
    else:
        text = compose(rng, ["swap {a} {f} on {c} to {t}", "swap {a} {f} to {t} on {c}", "convert {a} {f} on {c} into {t}", "swap {a} {f} for {t}, i hold it on {c}",
                             "i want to swap {a} {f} ({c}) to {t}"], a=a, f=fs, c=cs, t=ts)
        c.user(text)
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": a}
    r = c.call(("prepare_swap", args))
    assert r["status"] == "prepared_for_review", r
    c.say(f"I've prepared a swap of {a:g} {fs} on {CHAIN_LABEL[fc]}. You'd get {r['to']}, and the all-in cost is about ${r['total_cost_usd']:.2f}. "
          f"The route is {r['route']}. {rng.choice(['Review it and approve with Face ID when you are ready.', 'Nothing moves until you approve it with Face ID.'])}")
    calls = ([{"name": "get_balances", "arguments": {}}] if chainless else []) + [{"name": "prepare_swap", "arguments": args}]
    gold = {"tool": "prepare_swap", "name": None, "url": None, "calls": calls, "forbid_tools": ["open_app"],
            "answer": answer_gold(must_include=["NEAR Intents"], must_not_include_extra=["I opened"])}
    return finish(c, bio, "neg_swap", gold)


def sc_neg_deposit(rng):
    w, c, bio = new_conv(rng)
    sym = rng.choice(["USDC", "USDC", "USDC", "USDT", "ETH"])
    chain = rng.choice(["base", "arbitrum", "ethereum"])
    a = rng.choice([10, 20, 25, 50, 100, 200, 500, 1000]) if sym != "ETH" else rng.choice([0.05, 0.1, 0.25, 0.5, 1])
    w.balances = [b for b in w.balances if b["symbol"] != sym]
    w.ensure_balance(sym, chain, a)
    n, _ = written_name(rng, "Aave")
    text = compose(rng, ["deposit {a} {s} into {n}", "put {a} {s} in {n}", "supply {a} {s} to {n}", "lend {a} {s} on {n}", "i want to deposit {a} {s} into {n}",
                         "can you deposit {a} {s} into {n}", "stake {a} {s} in {n} for yield", "deposit {a} {s} in {n} pls", "earn on {a} {s} with {n}",
                         "add {a} {s} to {n}"], a=a, s=sym, n=n)
    c.user(text)
    args = {"project": "aave-v3", "symbol": sym, "amount": a}
    r = c.call(("prepare_deposit", args))
    assert r["status"] == "review_opened", r
    c.say(f"I've prepared a deposit of {r['amount']} into {r['pool']}, at about {r['apy_pct']}% APY right now. "
          f"{rng.choice(['Review it and approve with Face ID when you are ready.', 'Nothing moves until you approve it with Face ID.'])} The rate can change.")
    gold = {"tool": "prepare_deposit", "name": None, "url": None, "calls": [{"name": "prepare_deposit", "arguments": args}], "forbid_tools": ["open_app"],
            "answer": answer_gold(must_include=[r["pool"]], must_not_include_extra=["I opened"])}
    return finish(c, bio, "neg_deposit", gold)


def sc_neg_add_funds(rng):
    w, c, bio = new_conv(rng)
    if rng.random() < 0.75:
        goal = "wallet"
        text = compose(rng, ["how do I add funds", "how do i add money", "where do i deposit crypto", "how can i fund my wallet", "i want to add funds",
                             "how do i top up", "add funds", "show me how to deposit into my wallet", "where is my deposit address", "i need to put money in",
                             "how do i get crypto into hermit", "open add funds", "add money to my wallet pls"])
    else:
        goal = "hyperliquid"
        text = compose(rng, ["how do i fund my hyperliquid account", "i want to add funds to hyperliquid", "how do i deposit to hyperliquid", "add funds to my hyperliquid account",
                             "how can i put usdc on hyperliquid", "top up hyperliquid"])
    c.user(text)
    args = {"goal": goal}
    r = c.call(("open_add_funds", args))
    assert r["status"] == "button_shown", r
    where = "your wallet" if goal == "wallet" else "your Hyperliquid account"
    c.say(rng.choice([f"Tap the Add funds button below. It shows the deposit address and lets you pick a network for {where}.",
                      f"There's an Add funds button below. Tap it to see the deposit QR code and choose a network for {where}.",
                      f"Tap the Add funds button below to get the deposit address for {where}. Pick the network that matches where you're sending from."]))
    gold = {"tool": "open_add_funds", "name": None, "url": None, "calls": [{"name": "open_add_funds", "arguments": args}], "forbid_tools": ["open_app"],
            "answer": answer_gold(must_include=["Add funds"], must_not_include_extra=["I opened"])}
    return finish(c, bio, "neg_add_funds", gold)


def sc_neg_buy(rng):
    w, c, bio = new_conv(rng)
    sym = rng.choice(["WIF", "BONK", "JUP", "FARTCOIN", "TRUMP"])
    bare = False  # QA 2026-11-11: a bare "buy X" must ask for an amount (buy_jupiter), never default to $10
    usd = 10 if bare else rng.choice([5, 10, 15, 20, 25, 50, 100])
    w.balances = [b for b in w.balances if not (b["symbol"] == "USDC" and b["chain"] == "solana")]
    w.ensure_balance("USDC", "solana", usd)
    if bare:
        text = compose(rng, ["buy {s}", "buy some {s}", "i want to buy {s}", "get me some {s}", "pick up {s}", "ape into {s}", "buy {s} pls"], s=sym)
    else:
        text = compose(rng, ["buy ${u} of {s}", "buy {u} dollars of {s}", "get me ${u} worth of {s}", "i want to buy ${u} {s}", "ape ${u} into {s}",
                             "buy {s} with ${u}", "grab ${u} of {s}", "can you buy ${u} of {s} for me"], u=usd, s=sym)
    c.user(text)
    args = {"symbol": sym, "amount_usd": usd}
    r = c.call(("prepare_buy", args))
    assert r["status"] == "review_opened", r
    c.say(f"I've prepared a buy of {sym}. You pay {r['you_pay']} and get {r['you_get']} through Jupiter, with {r['slippage_pct']}% slippage allowed. "
          f"{'I used $' + str(usd) + ' because you gave no amount. ' if bare else ''}"
          f"{rng.choice(['Review it and approve with Face ID when you are ready.', 'Nothing moves until you approve it with Face ID.'])}")
    gold = {"tool": "prepare_buy", "name": None, "url": None, "calls": [{"name": "prepare_buy", "arguments": args}], "forbid_tools": ["open_app"],
            "answer": answer_gold(must_include=[sym], must_not_include_extra=["I opened"])}
    return finish(c, bio, "neg_buy", gold)


PAGES = {
    "Earn": (["the Earn page", "Earn", "the earn tab", "my earn page", "the earn screen", "earn"], "Tap Earn in the tab bar at the bottom."),
    "Trade": (["the Trade page", "the trade tab", "trade screen", "the trading page"], "Tap Trade in the tab bar at the bottom."),
    "Wallets": (["my wallets", "the wallets page", "wallets", "my wallet page", "the wallet screen", "my accounts"], "Tap Wallets in the tab bar at the bottom."),
    "Payments": (["the payments page", "payments", "the payments tab", "my payees"], "Tap Payments in the tab bar at the bottom."),
    "Settings": (["settings", "the settings page", "my settings", "app settings", "the settings screen"],
                 "Tap the menu button (three lines, top left) and then Settings."),
}
NAV_TPLS = ["open {p}", "open up {p}", "take me to {p}", "go to {p}", "pull up {p}", "show me {p}", "navigate to {p}", "switch to {p}", "bring up {p}",
            "can you open {p}", "i want to see {p}", "launch {p}", "get me to {p}", "open {p} for me", "head to {p}"]
NAV_PRE = ["I can't open pages in the app myself. ", "I can't switch pages from here. ", "I can't navigate the app for you. ", ""]


def sc_neg_nav(rng):
    w, c, bio = new_conv(rng)
    page = rng.choice(list(PAGES))
    ph, ans = PAGES[page]
    c.user(compose(rng, NAV_TPLS, p=rng.choice(ph)))
    pre = rng.choice(NAV_PRE)
    c.say(pre + ans)
    gold = {"tool": None, "name": None, "url": None, "calls": [], "text": True, "forbid_tools": ["open_app"],
            "answer": answer_gold(must_include=[page], must_not_include_extra=["I opened", "is open in"])}
    return finish(c, bio, "neg_nav", gold)


def sc_neg_real_site(rng):
    w, c, bio = new_conv(rng)
    name = rng.choice(CORE)
    right = rng.random() < 0.55
    if right:
        u = APPS[name]
        u = rng.choice(["", "", "https://"]) + u
    else:
        u = lookalike(rng, name)
    n, _ = written_name(rng, name)
    if rng.random() < 0.6:
        text = compose(rng, ["is {u} the real site", "is {u} the real {n} site", "is {u} legit", "is {u} the official {n}", "is this the real {n}? {u}",
                             "{u} is that the real one", "is {u} the right address for {n}", "is {u} safe to use", "can you check if {u} is the real {n}"],
                       allow_punct=False, u=u, n=n)
    else:
        text = compose(rng, ["someone sent me {u}, is that really {n}", "is {u} the actual {n} website"], allow_punct=False, u=u, n=n)
    c.user(text)
    listed = APPS[name]
    if domain(u) == domain(listed):
        a = rng.choice([f"That matches the address in Hermit's Explore list for {name}, which is {listed}. I can't check a site from here, though, so look at the address bar before you connect your wallet.",
                        f"Yes, {listed} is the address Hermit's Explore list uses for {name}, and that one matches. Still check the address bar before you connect your wallet, since I can't verify a site from here."])
    else:
        a = rng.choice([f"No. Hermit's Explore list uses {listed} for {name}, and that address is different. I can't verify a site from here, so don't connect your wallet there. Open {name} from Explore instead.",
                        f"That is not the address in Hermit's Explore list for {name}, which is {listed}. I can't prove a site is fake from here, but don't connect your wallet to it. Open {name} from Explore instead."])
    c.say(a)
    gold = {"tool": None, "name": None, "url": None, "calls": [], "text": True, "forbid_tools": ["open_app"],
            "answer": answer_gold(must_include=[listed], must_not_include_extra=["I opened"])}
    return finish(c, bio, "neg_real_site", gold, compare_name=name)


# =================================================================== assembly
POS = [(sc_open_name, 100), (sc_open_url, 43), (sc_open_url_lookalike, 14), (sc_bad_url, 14), (sc_not_found, 14), (sc_ask, 15), (sc_ask_then_give, 12)]
NEG = [(sc_neg_what_is, 18), (sc_neg_swap, 12), (sc_neg_deposit, 12), (sc_neg_add_funds, 10), (sc_neg_buy, 12), (sc_neg_nav, 18), (sc_neg_real_site, 8)]
EVAL_MIX = [(sc_open_name, 10), (sc_open_url, 4), (sc_open_url_lookalike, 2), (sc_bad_url, 2), (sc_not_found, 2), (sc_ask, 3), (sc_ask_then_give, 3),
            (sc_neg_what_is, 2), (sc_neg_swap, 1), (sc_neg_deposit, 1), (sc_neg_add_funds, 1), (sc_neg_buy, 1), (sc_neg_nav, 3), (sc_neg_real_site, 1)]


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def scaled(mix, n):
    tot = sum(k for _, k in mix)
    counts = [max(1, round(k * n / tot)) for _, k in mix]
    diff = n - sum(counts)
    i = 0
    while diff != 0:                 # fix rounding on the biggest bucket first
        j = sorted(range(len(mix)), key=lambda x: -mix[x][1])[i % len(mix)]
        step = 1 if diff > 0 else -1
        if counts[j] + step >= 1:
            counts[j] += step; diff -= step
        i += 1
    return [(fn, k) for (fn, _), k in zip(mix, counts)]


def build(seed, mix, n, exclude=()):
    rng = random.Random(seed)
    seen = set(exclude)
    rows = []
    for fn, k in scaled(mix, n):
        got = 0
        tries = 0
        while got < k:
            tries += 1
            assert tries < 400 * k + 400, f"cannot find {k} unique texts for {fn.__name__}"
            ex = fn(random.Random(rng.getrandbits(64)))
            key = norm(" || ".join(m["content"] for m in ex["messages"] if m["role"] == "user"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(ex)
            got += 1
    rng.shuffle(rows)
    return rows


def eval_record(ex, i):
    msgs = ex["messages"]
    j = max(k for k, m in enumerate(msgs) if m["role"] == "user")
    return {"id": f"{ex['meta']['kind']}-o{i}", "kind": ex["meta"]["kind"], "tools": True, "context": msgs[:j + 1], "reference": msgs[j + 1:],
            "expect": ex["meta"]["expect"], "world": ex["meta"]["world_final"], "gold": ex["meta"]["gold"],
            "checks": {"right_tool_new": {"expected_tool": ex["meta"]["gold"]["tool"], "calls": ex["meta"]["gold"]["calls"],
                                          "never_with": sorted(PREPARE_TOOLS) if ex["meta"]["gold"]["tool"] == "open_app" else ["open_app"]},
                      "exact_digits": {"url": ex["meta"]["gold"]["url"], "name": ex["meta"]["gold"]["name"]},
                      "asked_when_needed": bool(ex["meta"]["gold"].get("ask") or ex["meta"]["gold"].get("ask_after"))},
            "meta": {"source": "open_app_tool", "kind": ex["meta"]["kind"], "bio": ex["meta"]["bio"]}}


# =================================================================== self-check
NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def numbers(text):
    return {float(x.replace(",", "")) for x in NUM_RE.findall(text)}


def self_check(ex, where=""):
    msgs = ex["messages"]
    tag = f"{where} {ex['meta']['kind']}: {last_user(ex)!r}"
    json.loads(json.dumps(ex, ensure_ascii=False))
    assert ex["tools"] is True and ex["meta"]["source"] == "open_app_tool", tag
    assert msgs[0]["role"] == "system" and "{BIO}" not in msgs[0]["content"], tag
    bio = ex["meta"]["bio"]
    assert bio in ("Face ID", "Touch ID") and bio in msgs[0]["content"], tag
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    for m in msgs:
        assert other not in m["content"], f"{tag} wrong biometric word"
    users = [m["content"] for m in msgs if m["role"] == "user"]
    gold = ex["meta"]["gold"]
    # roles / tool pairing
    pending = 0
    call_list = []
    for m in msgs[1:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert m["content"] == "" and pending == 0, tag
            pending = len(m["tool_calls"])
            call_list += m["tool_calls"]
        elif m["role"] == "tool":
            pending -= 1
            json.loads(m["content"])
            assert pending >= 0, tag
        elif m["role"] == "assistant":
            assert pending == 0, tag
    assert pending == 0 and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"), tag
    # tool names / arguments vs schema
    for tc in call_list:
        assert tc["name"] in TOOL_SCHEMA, f"{tag} unknown tool {tc['name']}"
        sch = TOOL_SCHEMA[tc["name"]]
        for req in sch.get("required", []):
            assert req in tc["arguments"], (tag, tc, req)
        for k, v in tc["arguments"].items():
            assert k in sch["properties"], (tag, k)
            t = sch["properties"][k]["type"]
            assert (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) or (t == "string" and isinstance(v, str)), (tag, k, v)
            if "enum" in sch["properties"][k]:
                assert v in sch["properties"][k]["enum"], (tag, k, v)
    # calls after the LAST user turn match the gold exactly
    lu = max(k for k, m in enumerate(msgs) if m["role"] == "user")
    final_calls = [tc for m in msgs[lu + 1:] if m.get("tool_calls") for tc in m["tool_calls"]]
    assert final_calls == gold["calls"], f"{tag} calls {final_calls} != gold {gold['calls']}"
    assert [e for e in ex["meta"]["expect"] if "calls" in e] == [{"calls": [c]} for c in gold["calls"]] or \
        [tc for e in ex["meta"]["expect"] if "calls" in e for tc in e["calls"]] == gold["calls"], tag
    final_names = [tc["name"] for tc in final_calls]
    if gold["tool"] == "open_app":
        assert final_names.count("open_app") == 1 and not (set(final_names) & PREPARE_TOOLS), tag
    else:
        assert "open_app" not in final_names, tag
    # open_app arguments: name xor url, name in Explore keys, url verbatim in the user text
    for tc in call_list:
        if tc["name"] != "open_app":
            continue
        a = tc["arguments"]
        assert ("name" in a) != ("url" in a) and len(a) == 1, f"{tag} both/neither"
        if "name" in a:
            if gold.get("ask_after"):
                assert a["name"].lower() not in APPS_LOWER, tag
            else:
                assert a["name"].lower() in APPS_LOWER and a["name"] == APPS_LOWER[a["name"].lower()], f"{tag} name not in apps"
        else:
            assert any(a["url"] in u for u in users), f"{tag} url not verbatim"
            assert a["url"] == a["url"].strip() and a["url"][-1] not in ".,!?", tag
    # results vs answers
    tool_results = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    final = msgs[-1]["content"]
    low = final.lower()
    for b in BANNED:
        assert b not in low, f"{tag} banned phrase {b!r}"
    assert "\u00b7" not in final and final.strip(), tag
    corpus = " ".join(users + [m["content"] for m in msgs if m["role"] == "tool"] + [json.dumps(tc["arguments"]) for tc in call_list] +
                      [m["content"] for m in msgs[1:] if m["role"] == "assistant" and m is not msgs[-1]])
    cn = numbers(corpus)
    for x in numbers(final):
        assert x in cn, f"{tag} number {x} not from results/user text"
    assert numbers(final) is not None
    for kw in gold["answer"].get("must_include", []):
        assert kw.lower() in low, f"{tag} answer lacks {kw!r}"
    ma = gold["answer"].get("must_include_any")
    if ma:
        assert any(k.lower() in low for k in ma), f"{tag} answer names none of {ma}"
    for kw in gold["answer"].get("must_not_include_extra", []):
        assert kw.lower() not in low, f"{tag} answer includes {kw!r}"
    kind = ex["meta"]["kind"]
    opens = [r for r in tool_results if r.get("status") == "opened"]
    if opens:
        r = opens[0]
        label = final_calls[0]["arguments"].get("name")
        assert (label and label in final) or r["opened"] in final, f"{tag} answer must name what opened"
        assert NOTHING_LINE in final, tag
        # the address line: present exactly when a known dApp is named and the typed url's domain differs
        cmp = ex["meta"].get("compare_name")
        u = final_calls[0]["arguments"].get("url")
        want = bool(cmp and u and domain(u) != domain(APPS[cmp]))
        assert (CHECK_LINE in final) == want, f"{tag} check line {'missing' if want else 'unexpected'}"
        if label:
            assert CHECK_LINE not in final, tag
    if kind == "open_bad_url":
        assert tool_results[-1] == {"error": "Only http and https addresses can be opened."}, tag
        assert "http" in low and "https" in low and "opened" not in low.replace("open in", ""), tag
        if final_calls[0]["arguments"]["url"].lower().startswith("javascript:"):
            assert any(k in low for k in ("won't", "not going to", "can't", "sorry")), tag
    if kind == "open_not_found":
        assert tool_results[-1]["status"] == "not_found" and final.count("?") == 1 and "address" in low, tag
    if kind == "open_ask":
        assert not call_list and final.count("?") == 1 and "address" in low, tag
    if kind.startswith("neg_") and kind != "neg_nav":
        pass
    if kind == "neg_nav":
        assert not call_list and ("tab bar" in low or "menu" in low) and "can't open" not in low.replace("i can't open pages", ""), tag
    if kind in ("neg_swap", "neg_deposit", "neg_buy"):
        assert tool_results[-1].get("status") in ("prepared_for_review", "review_opened"), tag
        assert bio in final or "nothing moves" in low or "review" in low, tag
        assert not any(w in low for w in ("opened", "is open")), tag
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--eval-n", type=int, default=36)
    ap.add_argument("--eval-seed", type=int, default=1303)
    ap.add_argument("--out", default=os.path.join(LAB, "data3", "open_app_tool.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval3", "open_app_tool.jsonl"))
    a = ap.parse_args()
    assert a.eval_seed != a.seed
    train = build(a.seed, POS + NEG, a.n)
    train_keys = {norm(" || ".join(m["content"] for m in r["messages"] if m["role"] == "user")) for r in train}
    train_users = {norm(m["content"]) for r in train for m in r["messages"] if m["role"] == "user"}
    ev = build(a.eval_seed, EVAL_MIX, a.eval_n, exclude=train_keys)
    for r in train:
        self_check(r, "train")
    for r in ev:
        self_check(r, "eval")
    # duplicates and train/eval overlap, over the final user turn text and the full user text
    tk = [norm(last_user(r)) for r in train]
    ek = [norm(last_user(r)) for r in ev]
    for r in train:
        assert len({norm(m["content"]) for m in r["messages"] if m["role"] == "user"}) == len([m for m in r["messages"] if m["role"] == "user"])
    full_t = [norm(" || ".join(m["content"] for m in r["messages"] if m["role"] == "user")) for r in train]
    full_e = [norm(" || ".join(m["content"] for m in r["messages"] if m["role"] == "user")) for r in ev]
    assert len(set(full_t)) == len(full_t), "duplicate user texts in train"
    assert len(set(full_e)) == len(full_e), "duplicate user texts in eval"
    assert not (set(full_t) & set(full_e)), "eval text also in train"
    # disjoint final user turns (the thing being asked), except bare follow-up answers like "uniswap" in the ask-then-give kind
    for r in ev:
        if r["meta"]["kind"] != "open_ask_then_give":
            assert norm(last_user(r)) not in set(tk), f"eval user text in train: {last_user(r)!r}"
    assert len(set(ek)) == len(ek)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for i, r in enumerate(ev):
            f.write(json.dumps(eval_record(r, i), ensure_ascii=False) + "\n")
    bios = Counter(r["meta"]["bio"] for r in train)
    print(Counter(r["meta"]["kind"] for r in train))
    print(Counter(r["meta"]["kind"] for r in ev))
    print(bios)
    print(f"open_app_tool: {len(train)} train, {len(ev)} eval, self-check ok")


if __name__ == "__main__":
    main()
