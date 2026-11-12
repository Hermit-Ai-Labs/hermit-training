#!/usr/bin/env python3
"""get_quote conversations (the app's 12th tool, NEAR Intents quotes) for the Hermit 1.2 GPU retrain.

Teaches the split the app harness exposed:
  * "how much X would I get / quote / what's the rate"  -> get_quote, answer from the result, NO prepare
  * "swap / bridge / convert X to Y"                      -> prepare_swap (also to NEAR / bitcoin)
  * quote, then "ok do it"                                -> prepare_swap with the same arguments
  * missing source chain                                  -> one short question
  * unsupported route                                     -> explain, no retry loop
Every number in an answer comes from the tool result or the user.

  python3 build_quote.py --seed 1 --out ../data2/agent_quote.jsonl      (train)
  python3 build_quote.py --seed 999 --eval --out ../eval2/quote.jsonl   (held-out mini-suite)
"""
import argparse, json, os, random, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermit_fmt as H
import sim
from build_agent import Conv, S, DONE, amt_text

NATIVE_CHAIN = {"NEAR": "near", "SOL": "solana", "BTC": "bitcoin", "BNB": "bsc", "MON": "monad", "POL": "polygon"}
CHAIN_SAY = {"ethereum": ["Ethereum", "ethereum", "mainnet"], "base": ["Base", "base"], "arbitrum": ["Arbitrum", "arbitrum", "arb"],
             "bsc": ["BNB Chain", "BSC", "bsc"], "polygon": ["Polygon", "polygon"], "monad": ["Monad", "monad"],
             "solana": ["Solana", "solana"], "near": ["NEAR", "near"], "bitcoin": ["Bitcoin", "bitcoin"],
             "hyperliquid": ["Hyperliquid", "hyperliquid"]}
SRC = [("USDC", "base"), ("USDC", "arbitrum"), ("USDC", "ethereum"), ("USDT", "bsc"), ("USDC", "solana"), ("ETH", "base"),
       ("ETH", "arbitrum"), ("ETH", "ethereum"), ("SOL", "solana"), ("BNB", "bsc"), ("USDC", "polygon"), ("MON", "monad"),
       ("USDT", "ethereum"), ("NEAR", "near")]
DST = [("NEAR", "near"), ("SOL", "solana"), ("BTC", "bitcoin"), ("ETH", "ethereum"), ("ETH", "base"), ("ETH", "arbitrum"),
       ("USDC", "solana"), ("USDC", "base"), ("USDC", "hyperliquid"), ("BNB", "bsc"), ("USDC", "arbitrum"), ("MON", "monad")]


def say(rng, c):
    return rng.choice(CHAIN_SAY[c])


def pick_route(rng):
    while True:
        fs, fc = rng.choice(SRC); ts, tc = rng.choice(DST)
        if (fs, fc) != (ts, tc):
            return fs, fc, ts, tc


def pick_amount(rng, w, sym):
    p = w.prices[sym]
    usd = rng.choice([5, 10, 20, 25, 50, 100, 150, 200, 250, 500, 1000, 2000, 5000])
    a = usd / p
    if p > 1000:
        a = rng.choice([0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1])
    elif p > 50:
        a = rng.choice([0.1, 0.25, 0.5, 1, 2, 3, 5, 10])
    else:
        a = int(round(a)) if a >= 10 else round(a, 2)
    return a if a > 0 else 1.0


def dst_phrase(rng, ts, tc):
    if NATIVE_CHAIN.get(ts) == tc and rng.random() < 0.7:
        return ts                                      # "to NEAR", chain implied
    return f"{ts} on {say(rng, tc)}"


def eta(sec):
    return f"about {sec} seconds" if sec < 90 else f"about {round(sec / 60)} minutes"


def quote_answer(rng, r, offer=True):
    out = H.num(r["amount_out"], 8)
    base = rng.choice([
        f"{H.calc_str(r['amount_in'])} {r['from_symbol']} on {CHAIN_SAY[r['from_chain']][0]} gets you about {out} {r['to_symbol']} on {CHAIN_SAY[r['to_chain']][0]} ({S(r['amount_out_usd'])}), with {S(r['fee_usd'])} in fees and {eta(r['eta_seconds'])} to arrive.",
        f"The quote is {out} {r['to_symbol']} for {H.calc_str(r['amount_in'])} {r['from_symbol']}: {S(r['amount_in_usd'])} in, {S(r['amount_out_usd'])} out, fee {S(r['fee_usd'])}, {eta(r['eta_seconds'])}.",
        f"About {out} {r['to_symbol']} ({S(r['amount_out_usd'])}). Fees come to {S(r['fee_usd'])} and it takes {eta(r['eta_seconds'])}."])
    tail = rng.choice([" Quotes move with the market, so it can change a little. Want me to prepare it?",
                       " Want me to set it up for review?", " That's just a quote; nothing is prepared.", ""]) if offer else ""
    return base + tail


def sc_quote(rng):
    w = sim.World(rng); c = Conv(rng, w)
    fs, fc, ts, tc = pick_route(rng); a = pick_amount(rng, w, fs); at, _ = amt_text(rng, a)
    d = dst_phrase(rng, ts, tc)
    c.user(rng.choice([f"how much {d} would I get for {at} {fs} on {say(rng, fc)}?", f"quote {at} {fs} on {say(rng, fc)} to {d}",
                       f"what's the rate for {at} {fs} from {say(rng, fc)} to {d}?", f"if I swapped {at} {fs} on {say(rng, fc)} into {d}, how much would I get?",
                       f"price check: {at} {fs} ({say(rng, fc)}) -> {d}", f"how much is {at} {fs} on {say(rng, fc)} in {d}? just a quote",
                       f"get me a quote for bridging {at} {fs} from {say(rng, fc)} to {d}", f"what would {at} {fs} on {say(rng, fc)} turn into as {d}?"]))
    r = c.call(("get_quote", {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": a}))
    if "error" in r:
        c.say(quote_error_text(r["error"], fs, fc, ts, tc))
        return c.ex("quote_error")
    c.say(quote_answer(rng, r))
    return c.ex("quote")


def quote_error_text(e, fs, fc, ts, tc):
    if "destination only" in e and "bitcoin" in e:
        return "Bitcoin works only as a destination for these quotes, so I can't quote from it. I can quote the other way, into BTC."
    if "hyperliquid" in e:
        return "Hyperliquid only takes USDC as a destination. I can quote it into USDC on Hyperliquid instead."
    if "too low" in e:
        return f"That amount is too small to quote ({e.split(': ', 1)[-1]}). Try a slightly bigger amount."
    return f"I couldn't get a quote: {e}."


def sc_quote_ask(rng):
    """Source chain missing for a multi-chain token -> ask, then quote."""
    w = sim.World(rng); c = Conv(rng, w)
    fs = rng.choice(["ETH", "USDC", "USDT"]); fc = rng.choice({"ETH": ["ethereum", "base", "arbitrum"], "USDC": ["base", "arbitrum", "ethereum", "solana"], "USDT": ["bsc", "ethereum"]}[fs])
    ts, tc = rng.choice([("NEAR", "near"), ("SOL", "solana"), ("BTC", "bitcoin")])
    a = pick_amount(rng, w, fs); at, _ = amt_text(rng, a)
    c.user(rng.choice([f"quote {at} {fs} to {ts}", f"how much {ts} for {at} {fs}?", f"what's {at} {fs} worth in {ts}?"]))
    c.say(rng.choice([f"Which chain is the {fs} on?", f"Sure, which chain is your {fs} on? Base, Arbitrum, Ethereum or another?"]), ask=True)
    c.user(rng.choice([say(rng, fc), f"on {say(rng, fc)}", f"it's on {say(rng, fc)}"]))
    r = c.call(("get_quote", {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": a}))
    c.say(quote_answer(rng, r) if "error" not in r else quote_error_text(r["error"], fs, fc, ts, tc))
    return c.ex("quote_ask")


def sc_quote_then_prepare(rng):
    w = sim.World(rng); c = Conv(rng, w)
    fs, fc, ts, tc = pick_route(rng)
    while tc == "hyperliquid" or fc == "near":
        fs, fc, ts, tc = pick_route(rng)
    a = pick_amount(rng, w, fs); at, _ = amt_text(rng, a)
    c.user(rng.choice([f"how much {dst_phrase(rng, ts, tc)} would I get for {at} {fs} on {say(rng, fc)}?", f"quote {at} {fs} on {say(rng, fc)} to {dst_phrase(rng, ts, tc)}"]))
    r = c.call(("get_quote", {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": a}))
    if "error" in r:
        c.say(quote_error_text(r["error"], fs, fc, ts, tc)); return c.ex("quote_error")
    c.say(quote_answer(rng, r, offer=True) if rng.random() < 0.6 else quote_answer(rng, r, offer=False) + " Want me to prepare it?")
    c.user(rng.choice(["ok do it", "yes, swap it", "go ahead", "prepare it", "ok set it up", "yep"]))
    w.ensure_balance(fs, fc, a)
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": a}
    pr = c.call(("prepare_swap", args))
    if "error" in pr:
        c.say(f"I couldn't prepare it: {pr['error']}."); return c.ex("quote_prepare_error")
    c.say(f"I've prepared the swap of {at} {fs} on {CHAIN_SAY[fc][0]} to {ts} on {CHAIN_SAY[tc][0]}. The review screen shows the final amount and fees; {(lambda d: d[0].lower() + d[1:])(rng.choice(DONE))}")
    return c.ex("quote_then_prepare")


def sc_swap_direct(rng):
    """'swap/bridge X to NEAR/BTC/...': straight to prepare_swap, no quote first."""
    w = sim.World(rng); c = Conv(rng, w)
    fs, fc, ts, tc = pick_route(rng)
    while tc == "hyperliquid" or fc == "near":
        fs, fc, ts, tc = pick_route(rng)
    a = pick_amount(rng, w, fs); at, _ = amt_text(rng, a)
    verb = "bridge" if fs == ts else rng.choice(["swap", "convert", "trade"])
    c.user(rng.choice([f"{verb} {at} {fs} on {say(rng, fc)} to {dst_phrase(rng, ts, tc)}", f"{verb} {at} {fs} from {say(rng, fc)} into {dst_phrase(rng, ts, tc)}",
                       f"{verb} {at} {fs} ({say(rng, fc)}) for {dst_phrase(rng, ts, tc)}"]))
    w.ensure_balance(fs, fc, a)
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": a}
    pr = c.call(("prepare_swap", args))
    if "error" in pr:
        c.say(f"I couldn't prepare it: {pr['error']}."); return c.ex("swap_direct_error")
    c.say(f"I've prepared a {'bridge' if fs == ts else 'swap'} of {at} {fs} on {CHAIN_SAY[fc][0]} to {ts} on {CHAIN_SAY[tc][0]}. The review screen shows the amount you'll get and the fees; approve with Face ID to send it.")
    return c.ex("swap_direct")


INTENTS_QA = [
    ("what is NEAR Intents?", "NEAR Intents is a cross-chain swap network: you say what you want to end up with, and solvers compete to fill it, so you get one quote across chains instead of bridging and swapping by hand. Hermit uses it for quotes; you still approve any swap yourself."),
    ("is a quote the same as a swap?", "No. A quote just tells you roughly what you'd get, the fee and the time; nothing moves. A swap only happens after I prepare it and you approve it with Face ID."),
    ("why did my quote change?", "Quotes track live prices and liquidity, so they move by the second, more so for volatile tokens or big amounts. Ask again right before you swap and the review screen shows the final number."),
    ("can I quote into bitcoin?", "Yes, Bitcoin works as a destination for quotes, so you can quote USDC or ETH into BTC. It doesn't work as a source."),
]


def sc_intents_qa(rng):
    w = sim.World(rng); c = Conv(rng, w)
    q, a = rng.choice(INTENTS_QA)
    c.user(q); c.say(a)
    return c.ex("quote_notool")


TRAIN = [(sc_quote, 230), (sc_quote_ask, 50), (sc_quote_then_prepare, 90), (sc_swap_direct, 110), (sc_intents_qa, 24)]
EVAL = [(sc_quote, 16), (sc_quote_ask, 4), (sc_quote_then_prepare, 4), (sc_swap_direct, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for fn, n in (EVAL if a.eval else TRAIN):
        for _ in range(n):
            rows.append(fn(random.Random(rng.getrandbits(64))))
    rng.shuffle(rows)
    from build_agent import check
    for r in rows:
        check(r)
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            if a.eval:
                last_u = max(j for j, m in enumerate(r["messages"]) if m["role"] == "user")
                r = {"id": f"{r['meta']['kind']}-q{i}", "kind": r["meta"]["kind"], "tools": True, "context": r["messages"][:last_u + 1],
                     "reference": r["messages"][last_u + 1:], "expect": r["meta"]["expect"], "world": r["meta"]["world_final"]}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "->", a.out, Counter((r["meta"]["kind"] if not a.eval else None) for r in rows) if not a.eval else "")


if __name__ == "__main__":
    main()
