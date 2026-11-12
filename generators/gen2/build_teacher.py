#!/usr/bin/env python3
"""Breadth data from the local OPEN teacher (Qwen3-30B-A3B-Instruct-2507, Apache-2.0; no closed-API output).

Stages (each cached, re-runnable):
  questions : teacher writes realistic user questions per topic seed (topics below)
  answers   : teacher answers as Hermit (style rules + app facts sheet)
  multiturn : teacher plays the user for 3-6 turn conversations, Hermit answers each turn
  code      : trading/crypto code requests; python blocks must parse
Filters: finished generation, length cap (app max 640 tokens), no middle dots, no stray refusals,
no claimed live prices, simple arithmetic statements re-checked, no repeated sentences/paragraphs.
"""
import argparse, json, os, random, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher as T
import hermit_fmt as H

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = open(os.path.join(HERE, "app_facts.txt")).read()
OUT = os.path.join(HERE, "..", "data2")

# (topic id, description, n_questions, is_app_topic)
TOPICS = [
    ("perps_basics", "how perpetual futures work: margin, leverage, PnL, mark price, opening and closing", 90, False),
    ("funding", "perp funding rates: who pays whom, why, how often, annualising, funding arbitrage", 80, False),
    ("mark_oracle", "mark price vs oracle/index price vs last price, why liquidations use mark price", 50, False),
    ("liquidation", "liquidation on perps: how the price is found, maintenance margin, how to avoid it, partial liquidation", 90, False),
    ("cross_isolated", "cross margin vs isolated margin, pros and cons, which to use", 50, False),
    ("leverage_risk", "choosing leverage, why high leverage blows accounts up, effective leverage", 70, False),
    ("spot_vs_perps", "spot vs perps vs options for a normal trader, when to use each", 50, False),
    ("basis_carry", "basis trade, cash-and-carry, delta-neutral funding farming, risks", 60, False),
    ("sizing_risk", "position sizing, risk per trade, the 1% rule, account risk, correlation", 80, False),
    ("r_multiples", "R multiples, expectancy, win rate vs reward:risk, journaling trades", 60, False),
    ("stops", "stop-loss placement, stop hunting, trailing stops, take profit plans", 70, False),
    ("dca", "dollar-cost averaging, lump sum vs DCA, averaging down dangers", 50, False),
    ("orderbook", "reading an order book, depth, spread, liquidity, market vs limit orders, maker vs taker fees", 70, False),
    ("slippage_mev", "slippage, price impact, MEV, sandwich attacks and how to avoid them", 60, False),
    ("apy_apr", "APY vs APR, compounding, why advertised yields mislead", 50, False),
    ("il_lp", "liquidity providing, impermanent loss, concentrated liquidity ranges, LP fees", 70, False),
    ("stablecoins", "stablecoin types (fiat-backed, crypto-backed, synthetic like USDe), depeg risk, USDT vs USDC", 60, False),
    ("aave", "Aave lending and borrowing, supply APY, borrow APY, health factor, liquidation, LTV vs liquidation threshold", 90, False),
    ("defi_yield_risk", "where DeFi yield comes from, smart contract risk, points programs, real yield vs emissions", 60, False),
    ("prediction_markets", "prediction markets, reading odds as probabilities, expected value, Polymarket basics", 50, False),
    ("options", "options basics for crypto: calls, puts, premium, covered calls, implied volatility", 50, False),
    ("gas_evm", "gas on Ethereum and L2s: gwei, base fee, priority fee, why fails still cost gas, cheapest chains", 60, False),
    ("solana_fees", "Solana transactions: fees, priority fees, rent, failed transactions, token accounts", 40, False),
    ("bridges", "bridges and cross-chain transfers: how they work, risks, wrong-network deposits, wrapped tokens", 60, False),
    ("approvals", "token approvals, unlimited approvals, revoking approvals, Permit2 signatures, why approvals are risky", 70, False),
    ("scams", "crypto scam red flags: drainers, phishing sites, fake airdrops, address poisoning, honeypots, rug pulls, fake support DMs, pig butchering", 110, False),
    ("seed_safety", "seed phrase and private key safety, backups, what to do if a seed leaks, hardware wallets", 70, False),
    ("custody", "self-custody vs exchanges, counterparty risk, moving off exchanges, withdrawals and networks", 50, False),
    ("onchain_basics", "wallets, addresses, transactions, block explorers, token standards (ERC-20, SPL), checksummed addresses", 60, False),
    ("l2s", "Layer 2s (Base, Arbitrum), rollups, why they are cheaper, bridging to them", 40, False),
    ("hyperliquid", "Hyperliquid as a venue: perps, USDC margin, HYPE token, vaults like HLP, fees", 60, False),
    ("market_making", "market making basics: quoting, inventory risk, skew, adverse selection, spread capture", 60, False),
    ("psychology", "trading psychology: revenge trading, FOMO, tilt, sticking to a plan, taking breaks", 60, False),
    ("indicators", "technical analysis basics: support/resistance, moving averages, RSI, VWAP, volume, and their limits", 60, False),
    ("backtesting", "backtesting strategies, overfitting, lookahead bias, paper trading", 40, False),
    ("memecoins", "memecoins and launchpads: risks, liquidity, holder concentration, how people get rugged", 50, False),
    ("staking", "staking, liquid staking tokens, restaking, slashing, staking yield", 50, False),
    ("btc_eth", "Bitcoin and Ethereum fundamentals in plain language, halving, ETH burn, ETFs", 50, False),
    ("tax_basics", "general crypto tax concepts (capital gains, keeping records), without country-specific legal advice", 30, False),
    ("macro", "how macro (rates, CPI, dollar, liquidity) tends to affect crypto, in plain words", 40, False),
    ("portfolio", "portfolio construction for crypto: allocation, rebalancing, stablecoin buffer, diversification", 50, False),
    ("app_funding", "how to add funds to the Hermit wallet, QR deposits, which network is cheapest, withdrawing from an exchange to Hermit", 70, True),
    ("app_hyperliquid", "moving funds to Hyperliquid in Hermit, placing a perp trade in the app, TP/SL, reading the Positions tab, closing a position", 80, True),
    ("app_swaps", "swapping and bridging in Hermit with Relay, fees, what the review shows", 50, True),
    ("app_earn", "earning yield in Hermit with Aave v3, depositing and withdrawing, what APY means, risks", 60, True),
    ("app_safety", "how Hermit keeps the wallet safe: Secure Enclave, Face ID approvals, seed backup, daily limit, no iCloud", 60, True),
    ("app_capabilities", "what Hermit (the on-device AI) can and can't do, privacy, prepared actions and approvals, offline use", 60, True),
    ("app_troubleshoot", "troubleshooting in Hermit: deposit not showing, sent on wrong network, trade rejected for min size, insufficient margin, swap hidden", 50, True),
    ("writing", "everyday writing help: rewrite a message, summarise a paragraph, draft an email, make text more polite or shorter (include the text to work on in the question)", 120, False),
    ("general", "general everyday questions people ask an assistant: explanations of concepts, planning, advice, definitions (not crypto)", 120, False),
    ("money_life", "personal money questions: budgeting, emergency fund, debt, saving, explaining finance to family", 60, False),
]

CODE_TOPICS = [
    "a Python function for liquidation price of a perp position with maintenance margin",
    "a Python position-size calculator from account size, risk percent, entry and stop",
    "a Python script that fetches mid prices from the Hyperliquid info API (POST https://api.hyperliquid.xyz/info with {\"type\": \"allMids\"})",
    "a Python grid trading level generator", "a simple moving average crossover backtest in Python with pandas",
    "a Python function computing impermanent loss for a 50/50 pool", "a Python funding-rate annualiser and carry calculator",
    "a TypeScript function that formats token amounts from wei with decimals", "a Python RSI calculation without TA libraries",
    "a Python script that reads an ERC-20 balance via JSON-RPC eth_call", "a Python DCA simulator over a list of prices",
    "a Python Kelly criterion helper", "a Python function that validates an EIP-55 checksum address",
    "a Python max drawdown calculator for an equity curve", "a Python order book imbalance calculator",
    "a Python market maker quote function with inventory skew", "a Python trailing stop tracker",
    "a Python VWAP calculator from trades", "a Python script to compute PnL and ROE for a list of perp trades",
    "a Python Aave health factor calculator", "a JavaScript function to compute APY from APR and compounding periods",
    "a Python function to convert between bps, percent and decimals", "a Python Sharpe ratio calculator from daily returns",
    "a Solidity-free explanation plus Python code to estimate gas cost in USD", "a Python script to watch a wallet's ETH balance via RPC",
    "a Python function that splits a large order into TWAP slices", "a Python backtest of a breakout strategy with a stop",
    "a Python expected-value calculator for prediction market bets", "a Python script to rebalance a portfolio to target weights",
    "a Python function computing break-even price after fees for a round trip",
]

STYLE = """You are writing replies for Hermit, a private AI assistant that runs on the user's phone inside the Hermit app (a self-custody crypto wallet with Hyperliquid perps, Relay swaps and bridges, and Aave earn).
Write exactly the reply Hermit would give. Rules:
- Short, plain and direct. Lead with the answer. Usually 2 to 6 sentences; use a short list only for steps or comparisons. At most about 180 words unless the user asks for code or a long piece of writing.
- Simple words. No filler, no greetings, no "Great question", no sign-offs. No "not financial advice" or "consult a professional" boilerplate, no moralising. Give a real opinion when asked.
- Never use the middle dot character.
- Be honest: if something depends on live data (today's price, current funding, the user's balance) or you are not sure, say so plainly instead of guessing. Never state a current price or rate as fact.
- Maths: if you use numbers, keep examples simple and double-check every calculation.
- Help with all legitimate trading, crypto, coding and money questions. Warn clearly about scams. Refuse only to help build drainers, phishing or scam kits, malware, or anything sexual involving minors, in one plain sentence.
- Only describe app features from the facts below; never invent features. If the app can't do something, say so.

APP FACTS:
""" + FACTS


QGEN = """Write {k} different questions that real users of a crypto wallet + trading app with a built-in AI assistant would type to the assistant about this topic:
TOPIC: {topic}
Mix: beginners and experienced traders; short casual messages (some lowercase, some with small typos) and fuller questions; "how", "why", "what if", "should I", comparisons, and concrete scenarios with their own numbers. Each question must make sense on its own and must be answerable WITHOUT live data (do not ask for today's price or the user's balance). {extra}
Return ONLY a JSON array of strings."""
PERSONAS = ["Lean towards beginners.", "Lean towards experienced perp traders.", "Lean towards DeFi users.",
            "Include a few with specific numbers.", "Make several very short (under 8 words).", ""]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def parse_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [x.strip() for x in arr if isinstance(x, str) and 6 < len(x.strip()) < 600]
    except Exception:
        return []


# ---------------------------------------------------------------- filters
REFUSE = re.compile(r"\b(I can(?:'|no)t (?:help|assist|provide)|I'm (?:not able|unable) to (?:help|assist)|I won't (?:help|provide))", re.I)
LIVEPRICE = re.compile(r"(currently|right now|today)[^.\n]{0,40}(trading at|priced at|is at|sits at) \$?\d", re.I)
BOILER = re.compile(r"(not financial advice|consult (?:a|with a) (?:financial|professional|tax)|as an ai\b|I'm just an AI|Great question)", re.I)
ARITH = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([x×*/+\-])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)")


def arith_ok(text):
    for a, op, b, c in ARITH.findall(text):
        try:
            a_, b_, c_ = (float(v.replace(",", "")) for v in (a, b, c))
        except ValueError:
            continue
        v = {"x": a_ * b_, "×": a_ * b_, "*": a_ * b_, "/": a_ / b_ if b_ else None, "+": a_ + b_, "-": a_ - b_}[op]
        if v is None:
            return False
        if abs(v - c_) > max(0.011, abs(c_) * 0.006):
            return False
    return True


def sentences(t):
    return [H.re.sub(r"\s+", " ", s).strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s.strip()) >= 25]


def clean(text, allow_long=False, allow_refusal=False):
    t = text.strip()
    if not t:
        return None
    t = t.replace(" · ", ", ").replace("·", ",")
    t = re.sub(r"^(Hermit:|\*\*Hermit:\*\*)\s*", "", t)
    if BOILER.search(t) or LIVEPRICE.search(t):
        return None
    if not allow_refusal and REFUSE.search(t[:200]):
        return None
    if len(t) > (2600 if allow_long else 1500):
        return None
    if not arith_ok(t):
        return None
    ss = sentences(t)
    if len(ss) != len(set(ss)):
        return None
    if re.search(r"[\U0001F300-\U0001FAFF]", t):
        t = re.sub(r"[\U0001F300-\U0001FAFF]\s?", "", t)
    return t


def code_ok(t):
    blocks = re.findall(r"```(\w*)\n(.*?)```", t, re.S)
    if not blocks:
        return False
    import ast
    for lang, code in blocks:
        if lang.lower() in ("python", "py", ""):
            try:
                ast.parse(code)
            except SyntaxError:
                return False
    return True


# ---------------------------------------------------------------- stages
def stage_questions(rng, scale):
    jobs, meta = [], []
    for tid, desc, n, app in TOPICS:
        n = max(10, int(n * scale))
        calls = max(1, round(n / 22))
        for i in range(calls + 1):        # one extra call for headroom after dedup
            jobs.append({"messages": [{"role": "user", "content": QGEN.format(k=25, topic=desc, extra=PERSONAS[i % len(PERSONAS)])}],
                         "temperature": 0.95, "max_tokens": 1400, "seed": 100 + i})
            meta.append((tid, n))
    res = T.map_chat(jobs, workers=16, log_every=50)
    byt, seen = {}, set()
    for (tid, n), r in zip(meta, res):
        for q in parse_list(r["text"]):
            k = norm(q)
            if k in seen:
                continue
            seen.add(k)
            byt.setdefault(tid, []).append(q)
    out = []
    for tid, desc, n, app in TOPICS:
        qs = byt.get(tid, [])
        rng.shuffle(qs)
        out += [{"topic": tid, "app": app, "q": q} for q in qs[:max(10, int(n * scale))]]
    return out


def answer_jobs(qs):
    return [{"messages": [{"role": "system", "content": STYLE}, {"role": "user", "content": q["q"]}],
             "temperature": 0.6, "max_tokens": 650, "seed": 7} for q in qs]


def stage_answers(qs):
    res = T.map_chat(answer_jobs(qs), workers=16, log_every=200)
    out, drop = [], Counter()
    for q, r in zip(qs, res):
        if r["finish"] != "stop":
            drop["unfinished"] += 1; continue
        t = clean(r["text"], allow_long=q["topic"] == "writing")
        if t is None:
            drop["filtered"] += 1; continue
        out.append({"topic": q["topic"], "app": q["app"], "q": q["q"], "a": t})
    print("answers kept", len(out), "dropped", dict(drop))
    return out


USER_SIM = """You are role-playing a user of the Hermit app chatting with its AI assistant. Here is the conversation so far:
{conv}
Write the user's NEXT message only: a natural, short follow-up (usually under 25 words) that builds on the last answer: ask for clarification, push back, give their own numbers, change angle slightly, or ask a related practical question. Don't thank or say goodbye. Output only the message text."""


def stage_multiturn(seeds, rng, n_conv):
    convs = [{"topic": s["topic"], "msgs": [{"role": "user", "content": s["q"]}, {"role": "assistant", "content": s["a"]}],
              "target": rng.choice([3, 3, 4, 4, 5, 6])} for s in rng.sample(seeds, min(n_conv, len(seeds)))]
    for turn in range(2, 7):
        live = [c for c in convs if len(c["msgs"]) // 2 < c["target"] and not c.get("dead")]
        if not live:
            break
        print(f"multiturn turn {turn}: {len(live)} live", flush=True)
        uj = [{"messages": [{"role": "user", "content": USER_SIM.format(conv="\n".join(f"{m['role'].upper()}: {m['content']}" for m in c["msgs"]))}],
               "temperature": 0.9, "max_tokens": 80, "seed": turn} for c in live]
        ur = T.map_chat(uj, workers=16, log_every=200)
        for c, r in zip(live, ur):
            u = r["text"].strip().strip('"')
            if r["finish"] != "stop" or not (3 < len(u) < 300) or "\n" in u.strip():
                c["dead"] = True; continue
            c["msgs"].append({"role": "user", "content": u})
        live = [c for c in live if not c.get("dead")]
        aj = [{"messages": [{"role": "system", "content": STYLE + "\nThis is a multi-turn chat: don't repeat what you already said; build on it and stay consistent."}] + c["msgs"],
               "temperature": 0.6, "max_tokens": 600, "seed": turn} for c in live]
        ar = T.map_chat(aj, workers=16, log_every=200)
        for c, r in zip(live, ar):
            t = clean(r["text"]) if r["finish"] == "stop" else None
            if t is None:
                c["msgs"].pop(); c["dead"] = True; continue
            prev = set(x for m in c["msgs"] if m["role"] == "assistant" for x in sentences(m["content"]))
            if prev & set(sentences(t)):
                c["msgs"].pop(); c["dead"] = True; continue       # repeated a sentence from an earlier turn
            c["msgs"].append({"role": "assistant", "content": t})
    out = [c for c in convs if len(c["msgs"]) >= 6]
    print("multiturn kept", len(out), "of", len(convs))
    return out


CODE_REQ = """Write one realistic request a trader would send to an AI assistant asking for code: {t}. Vary the wording and add one specific detail or constraint. Output only the request."""


def stage_code(rng, n):
    jobs = [{"messages": [{"role": "user", "content": CODE_REQ.format(t=rng.choice(CODE_TOPICS))}], "temperature": 1.0, "max_tokens": 120, "seed": i}
            for i in range(n)]
    reqs = [r["text"].strip().strip('"') for r in T.map_chat(jobs, workers=16, log_every=200)]
    reqs = list({norm(r): r for r in reqs if 15 < len(r) < 500}.values())
    aj = [{"messages": [{"role": "system", "content": STYLE + "\nFor code: give compact, correct, runnable code in one block (no placeholders for logic), then one or two lines on how to use it. Keep the whole reply under 450 words."},
                        {"role": "user", "content": q}], "temperature": 0.4, "max_tokens": 900, "seed": 3} for q in reqs]
    out = []
    for q, r in zip(reqs, T.map_chat(aj, workers=16, log_every=200)):
        t = r["text"].strip().replace("·", ",")
        if r["finish"] == "stop" and code_ok(t) and len(t) < 2600 and not BOILER.search(t):
            out.append({"topic": "code", "app": False, "q": q, "a": t})
    print("code kept", len(out), "of", len(reqs))
    return out


def dump(rows, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "->", p, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["questions", "answers", "multiturn", "code", "all"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n-multi", type=int, default=1400)
    ap.add_argument("--n-code", type=int, default=900)
    a = ap.parse_args()
    rng = random.Random(42)
    if a.stage in ("questions", "all"):
        dump(stage_questions(rng, a.scale), "t_questions.jsonl")
    if a.stage in ("answers", "all"):
        qs = [json.loads(l) for l in open(os.path.join(OUT, "t_questions.jsonl"))]
        dump(stage_answers(qs), "t_answers.jsonl")
    if a.stage in ("code", "all"):
        dump(stage_code(rng, a.n_code), "t_code.jsonl")
    if a.stage in ("multiturn", "all"):
        seeds = [json.loads(l) for l in open(os.path.join(OUT, "t_answers.jsonl"))]
        seeds = [s for s in seeds if s["topic"] not in ("writing",)]
        dump(stage_multiturn(seeds, rng, a.n_multi), "t_multiturn.jsonl")
