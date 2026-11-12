#!/usr/bin/env python3
"""Trading / DeFi maths generator v2 for Hermit 1.2. Every answer is computed here, from the numbers
exactly as they appear in the question (v1 computed a few answers from unrounded values).

Each item: {category, question, answer, tol, unit, expr, steps}
  expr  : a calculator expression whose value IS the answer (verified at build time)
  steps : short worked steps for the no-tool answer (every number in it computed here)
"""
import random
from hermit_fmt import calc, money, num, pct, smart_money


def P(rng, bands=((0.5, 5), (5, 50), (50, 500), (500, 5000), (5000, 90000))):
    lo, hi = rng.choice(bands)
    p = rng.uniform(lo, hi)
    return round(p, 2) if p < 1000 else float(round(p))


def fp(x):
    """Plain number as typed in a question/expression (no thousands separator)."""
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.10g}"


C = {}


def cat(name, unit, tol):
    def deco(f):
        C[name] = (f, unit, tol)
        return f
    return deco


JN = ["Just the number.", "Answer with just the number.", "Only the number please.", ""]


def tail(rng, extra=""):
    t = rng.choice(JN)
    return (" " + (extra + " " if extra else "") + t).rstrip()


# ---------------------------------------------------------------- v1 categories (fixed)
@cat("pnl_long", "usd", 0.01)
def g_pnl_long(rng):
    e = P(rng); mv = rng.choice([3, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]) / 100
    x = round(e * (1 + mv if rng.random() < 0.7 else 1 - mv), 2); q = rng.choice([0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 25, 100])
    ans = (x - e) * q
    qs = rng.choice([f"I went long {fp(q)} units at ${fp(e)} and closed at ${fp(x)}. What is my profit in dollars?{tail(rng)}",
                     f"Bought {fp(q)} at ${fp(e)}, sold at ${fp(x)}. How much did I make or lose?",
                     f"Long {fp(q)} coins from {fp(e)} to {fp(x)}, what's the PnL in USD?"])
    return qs, ans, f"({fp(x)} - {fp(e)}) * {fp(q)}", f"PnL = (exit - entry) x size = ({fp(x)} - {fp(e)}) x {fp(q)} = {smart_money(ans)}."


@cat("pnl_short", "usd", 0.01)
def g_pnl_short(rng):
    e = P(rng); mv = rng.choice([3, 4, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30]) / 100
    x = round(e * (1 - mv if rng.random() < 0.7 else 1 + mv), 2); q = rng.choice([0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 50])
    ans = (e - x) * q
    qs = rng.choice([f"I shorted {fp(q)} units at ${fp(e)} and covered at ${fp(x)}. Profit in dollars (negative if a loss)?{tail(rng)}",
                     f"Short {fp(q)} from {fp(e)}, closed at {fp(x)}. What's my PnL?",
                     f"I opened a {fp(q)} unit short at ${fp(e)} and bought it back at ${fp(x)}. Did I make money, and how much?"])
    return qs, ans, f"({fp(e)} - {fp(x)}) * {fp(q)}", f"Short PnL = (entry - exit) x size = ({fp(e)} - {fp(x)}) x {fp(q)} = {smart_money(ans)}."


@cat("liq_price_long", "usd", 0.01)
def g_liq_long(rng):
    e = P(rng); lev = rng.choice([2, 3, 4, 5, 8, 10, 15, 20, 25, 50, 100]); ans = e * (1 - 1 / lev)
    qs = rng.choice([f"I am long {lev}x with entry at ${fp(e)}. Ignoring fees and maintenance margin, at what price am I liquidated?{tail(rng)}",
                     f"Where's my liquidation on a {lev}x long from {fp(e)}? Ignore fees and maintenance margin.",
                     f"{lev}x long entered at ${fp(e)}, no maintenance margin. Liq price?"])
    return qs, ans, f"{fp(e)} * (1 - 1/{lev})", f"A {lev}x long is wiped out by a 1/{lev} = {pct(100/lev)} drop. Liq = {fp(e)} x (1 - 1/{lev}) = {smart_money(ans)}."


@cat("liq_price_short", "usd", 0.01)
def g_liq_short(rng):
    e = P(rng); lev = rng.choice([2, 3, 4, 5, 8, 10, 15, 20, 25, 50]); ans = e * (1 + 1 / lev)
    qs = rng.choice([f"I am short {lev}x with entry at ${fp(e)}. Ignoring fees and maintenance, at what price am I liquidated?{tail(rng)}",
                     f"Short at {fp(e)} with {lev}x leverage. Where do I get liquidated, ignoring maintenance margin?",
                     f"{lev}x short from ${fp(e)}: liquidation price with no maintenance margin?"])
    return qs, ans, f"{fp(e)} * (1 + 1/{lev})", f"A {lev}x short is wiped out by a 1/{lev} = {pct(100/lev)} rise. Liq = {fp(e)} x (1 + 1/{lev}) = {smart_money(ans)}."


@cat("fee", "usd", 0.01)
def g_fee(rng):
    n = round(rng.uniform(500, 200000), 2); b = rng.choice([1, 1.5, 2, 2.5, 4.5, 5, 7, 7.5, 10, 15, 25, 30, 50]); ans = n * b / 10000
    qs = rng.choice([f"A fee of {fp(b)} bps applies to a {money(n)} order. Fee in dollars?{tail(rng)}",
                     f"What's the fee on a ${fp(n)} trade at {fp(b)} basis points?",
                     f"Taker fee is {fp(b)} bps. I trade {money(n)} notional. How much do I pay?"])
    return qs, ans, f"{fp(n)} * {fp(b)} / 10000", f"{fp(b)} bps = {fp(b)}/10000. Fee = {fp(n)} x {fp(b)}/10000 = {smart_money(ans)}."


@cat("spread_bps", "bps", 0.05)
def g_spread(rng):
    mid = P(rng); h = rng.choice([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.5]) * mid / 100
    bid, ask = round(mid - h, 2), round(mid + h, 2); mid = round((bid + ask) / 2, 3)
    if ask <= bid:
        ask = round(bid + 0.01, 2); mid = round((bid + ask) / 2, 3)
    ans = (ask - bid) / mid * 10000
    qs = rng.choice([f"Bid is {fp(bid)}, ask is {fp(ask)}. What is the spread in basis points off the mid of {fp(mid)}?{tail(rng)}",
                     f"Order book shows {fp(bid)} bid and {fp(ask)} ask (mid {fp(mid)}). How wide is the spread in bps?"])
    return qs, ans, f"({fp(ask)} - {fp(bid)}) / {fp(mid)} * 10000", f"Spread = {fp(ask)} - {fp(bid)} = {num(ask-bid, 4)}. In bps = {num(ask-bid,4)}/{fp(mid)} x 10000 = {num(ans, 2)} bps."


@cat("funding_annual", "percent", 0.05)
def g_fund_ann(rng):
    r = round(rng.choice([1, -1]) * rng.uniform(0.005, 0.15), 4); hrs = rng.choice([8, 8, 1])
    per = 24 // hrs * 365; ans = r * per
    qs = rng.choice([f"Perp funding is {fp(r)}% every {hrs} hour{'s' if hrs > 1 else ''}. Annualised simply (no compounding), what percent per year?{tail(rng)}",
                     f"Funding is {fp(r)}% per {hrs}h. What's that as a simple yearly rate?"])
    return qs, ans, f"{fp(r)} * {24 // hrs} * 365", f"{24 // hrs} periods a day x 365 = {per} periods. {fp(r)}% x {per} = {pct(ans)} a year."


@cat("pct_roundtrip", "percent", 0.05)
def g_round(rng):
    a = rng.randint(5, 90); b = rng.randint(5, 150); f = (1 - a / 100) * (1 + b / 100); ans = abs(1 - f) * 100
    word = "down" if f < 1 else "up"
    qs = rng.choice([f"A token drops {a}% then rises {b}%. Overall, what percent is it {word} from the start? Just the magnitude as a number.",
                     f"My coin fell {a}% and then pumped {b}%. Am I up or down overall, and by how much?"])
    return qs, ans, f"abs((1 - {a}/100) * (1 + {b}/100) - 1) * 100", f"Multiplier = (1 - {a/100:g}) x (1 + {b/100:g}) = {num(f, 4)}. That is {pct(ans)} {word} from the start."


@cat("breakeven_fees", "usd", 0.02)
def g_be(rng):
    e = P(rng); b = rng.choice([4, 5, 6, 8, 9, 10, 12, 15, 20, 25, 30, 40, 60]); ans = e * (1 + b / 10000)
    qs = rng.choice([f"I buy at ${fp(e)}. Round-trip fees total {b} bps. At what sell price do I break even?{tail(rng)}",
                     f"Entry {fp(e)}, total fees in and out are {b} bps. Break-even exit price?"])
    return qs, ans, f"{fp(e)} * (1 + {b}/10000)", f"Need to cover {b} bps. Break-even = {fp(e)} x (1 + {b}/10000) = {smart_money(ans)}."


@cat("position_size", "units", 0.001)
def g_ps(rng):
    acct = rng.choice([500, 1000, 2500, 5000, 7500, 10000, 15000, 25000, 50000, 100000]); rp = rng.choice([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5])
    e = P(rng); sp = rng.choice([1, 2, 3, 4, 5, 7, 8, 10, 15]); risk = acct * rp / 100; dist = e * sp / 100; ans = risk / dist
    qs = rng.choice([f"My account is {money(acct)}. I risk {fp(rp)}% per trade. Entry ${fp(e)}, stop {sp}% away. How many units should I buy so the stop loses exactly my risk? Round to 4dp, just the number.",
                     f"Account {money(acct)}, risking {fp(rp)}% a trade. Entry {fp(e)} with a stop {sp}% below. Position size in coins?"])
    return qs, ans, f"({fp(acct)} * {fp(rp)} / 100) / ({fp(e)} * {sp} / 100)", f"Risk = {money(acct)} x {fp(rp)}% = {money(risk)}. Stop distance = {fp(e)} x {sp}% = {money(dist, 4)}. Size = {money(risk)} / {money(dist, 4)} = {num(ans, 4)} units."


@cat("amm_v2_out", "usdc", 0.1)
def g_amm(rng):
    x = rng.choice([50, 100, 200, 300, 500, 750, 1000, 2000]); y = x * rng.choice([1500, 1800, 2000, 2200, 2500, 3000, 3500])
    a = rng.choice([0.5, 1, 1.5, 2, 3, 5, 10, 15, 20]); af = a * 0.997; ans = y * af / (x + af)
    qs = rng.choice([f"A Uniswap v2 pool has {x} ETH and {y:,} USDC. With the 0.3% fee, how much USDC do I get for {fp(a)} ETH? Round to 2dp, just the number.",
                     f"Constant product pool: {x} ETH / {y} USDC, 0.3% fee. What do I receive selling {fp(a)} ETH?"])
    return qs, ans, f"{y} * ({fp(a)} * 0.997) / ({x} + {fp(a)} * 0.997)", f"In after fee = {fp(a)} x 0.997 = {num(af, 4)}. Out = y x in / (x + in) = {y} x {num(af,4)} / ({x} + {num(af,4)}) = {num(ans, 2)} USDC."


@cat("impermanent_loss", "percent", 0.05)
def g_il(rng):
    p = round(rng.choice([rng.uniform(1.1, 5), rng.uniform(0.2, 0.9)]), 2); il = 2 * p ** 0.5 / (1 + p) - 1; ans = abs(il) * 100
    qs = rng.choice([f"A token in a 50/50 LP moves {fp(p)}x against the other. What is the impermanent loss as a percent (magnitude)? Round to 2dp, just the number.",
                     f"I LP a 50/50 pool and one side's price changes by a factor of {fp(p)}. How much impermanent loss is that?"])
    return qs, ans, f"abs(2 * sqrt({fp(p)}) / (1 + {fp(p)}) - 1) * 100", f"IL = 2 x sqrt(r) / (1 + r) - 1 with r = {fp(p)}: 2 x {num(p**0.5, 4)} / {num(1+p, 4)} - 1 = {num(il, 4)}, so {pct(ans)}."


@cat("mm_spread_capture", "usd", 0.01)
def g_mm(rng):
    b = rng.choice([1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30]); s = rng.choice([500, 1000, 2500, 5000, 10000, 20000, 50000, 100000])
    t = rng.choice([20, 50, 75, 100, 150, 200, 300, 500, 1000]); ans = b / 10000 * s * t
    qs = rng.choice([f"I make markets quoting {b} bps wide and capture the full spread on {t} round trips of {money(s)} each. Gross profit in dollars?{tail(rng)}",
                     f"Market making: {b} bps captured per round trip, {t} round trips of ${s}. Gross?"])
    return qs, ans, f"{b} / 10000 * {s} * {t}", f"Per trip = {b} bps x {money(s)} = {money(b/10000*s)}. Times {t} trips = {smart_money(ans)}."


@cat("sum_balances", "token", 0.001)
def g_sum(rng):
    n = rng.choice([3, 4, 5]); v = [round(rng.uniform(0.1, 5), rng.choice([1, 2])) for _ in range(n)]; tok = rng.choice(["SOL", "ETH", "BNB", "HYPE"])
    ans = sum(v)
    qs = rng.choice([f"I have {n} wallets with {', '.join(fp(x) for x in v)} {tok}. Total {tok}? Round to 2dp, just the number.",
                     f"Add up my {tok}: {', '.join(fp(x) for x in v)}."])
    return qs, ans, " + ".join(fp(x) for x in v), f"{' + '.join(fp(x) for x in v)} = {num(ans, 2)} {tok}."


@cat("kelly", "percent", 0.05)
def g_kelly(rng):
    pi = rng.randint(45, 80); p = pi / 100; b = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5]); f = p - (1 - p) / b; ans = f * 100
    qs = rng.choice([f"A strategy wins {pi}% of the time, paying {fp(b)}:1 (win {fp(b)} units, lose 1). What fraction of the bankroll does the Kelly criterion suggest, as a percent? Round to 2dp, just the number.",
                     f"Win rate {pi}%, payoff {fp(b)} to 1. Full Kelly bet size in percent?"])
    return qs, ans, f"({fp(p)} - (1 - {fp(p)}) / {fp(b)}) * 100", f"Kelly f = p - (1 - p) / b = {fp(p)} - {num(1-p, 2)}/{fp(b)} = {num(f, 4)}, so {pct(ans)} of the bankroll{' (negative means no edge, bet nothing)' if f < 0 else ''}."


@cat("option_payoff", "usd", 0.01)
def g_opt(rng):
    k = P(rng); pr = round(k * rng.uniform(0.005, 0.05), 2); s = round(k * rng.choice([1.03, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3, 0.85, 0.9, 0.92, 0.95, 0.97]), 2)
    call = rng.random() < 0.6; intr = max(s - k, 0) if call else max(k - s, 0); ans = intr - pr; kind = "call" if call else "put"
    qs = rng.choice([f"I bought a ${fp(k)} {kind} for a premium of ${fp(pr)}. At expiry the spot is ${fp(s)}. Profit per contract in dollars (negative if a loss)?{tail(rng)}",
                     f"Long a {fp(k)} strike {kind}, paid {fp(pr)}. Expiry price {fp(s)}. PnL per contract?"])
    e = f"max({fp(s)} - {fp(k)}, 0) - {fp(pr)}" if call else f"max({fp(k)} - {fp(s)}, 0) - {fp(pr)}"
    return qs, ans, e, f"Intrinsic = max({'spot - strike' if call else 'strike - spot'}, 0) = {money(intr)}. Profit = {money(intr)} - {money(pr)} = {smart_money(ans)}."


@cat("dca", "usd", 0.01)
def g_dca(rng):
    a = rng.choice([10, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 500]); w = rng.choice([4, 8, 12, 13, 26, 39, 52, 104]); ans = a * w
    qs = rng.choice([f"If I DCA {money(a)} every week for {w} weeks, how much do I invest in total?{tail(rng)}",
                     f"${a} a week for {w} weeks, total invested?"])
    return qs, ans, f"{a} * {w}", f"{money(a)} x {w} weeks = {smart_money(ans)}."


# ---------------------------------------------------------------- new categories
@cat("liq_mm", "usd", 0.01)
def g_liq_mm(rng):
    side = rng.choice(["long", "short"]); e = P(rng); lev = rng.choice([3, 5, 10, 20, 25, 40]); mm = rng.choice([0.4, 0.5, 1, 1.25, 2]) / 100
    ans = e * (1 - 1 / lev) / (1 - mm) if side == "long" else e * (1 + 1 / lev) / (1 + mm)
    qs = f"Isolated {lev}x {side} from ${fp(e)}, maintenance margin {fp(mm*100)}%. Liquidation price? Use liq = entry x (1 {'-' if side=='long' else '+'} 1/lev) / (1 {'-' if side=='long' else '+'} mm).{tail(rng)}"
    if rng.random() < 0.5:
        qs = f"Where's the liquidation price on a {lev}x {side} at {fp(e)} if maintenance margin is {fp(mm*100)}%?"
    e_ = f"{fp(e)} * (1 - 1/{lev}) / (1 - {fp(mm)})" if side == "long" else f"{fp(e)} * (1 + 1/{lev}) / (1 + {fp(mm)})"
    return qs, ans, e_, f"Liq = entry x (1 {'-' if side=='long' else '+'} 1/lev) / (1 {'-' if side=='long' else '+'} mm) = {fp(e)} x (1 {'-' if side=='long' else '+'} 1/{lev}) / (1 {'-' if side=='long' else '+'} {fp(mm)}) = {smart_money(ans)}."


@cat("roe", "percent", 0.05)
def g_roe(rng):
    side = rng.choice(["long", "short"]); e = P(rng); lev = rng.choice([2, 3, 5, 10, 15, 20, 25]); mv = rng.choice([1, 1.5, 2, 3, 4, 5, 6, 8]) / 100
    x = round(e * (1 + mv * rng.choice([1, -1])), 2); sgn = 1 if side == "long" else -1; ans = sgn * (x - e) / e * lev * 100
    qs = rng.choice([f"I'm {lev}x {side} from {fp(e)} and price is now {fp(x)}. What's my ROE in percent (return on margin)?{tail(rng)}",
                     f"{side.title()} at ${fp(e)}, {lev}x leverage, mark ${fp(x)}. ROE %?"])
    return qs, ans, f"{sgn} * ({fp(x)} - {fp(e)}) / {fp(e)} * {lev} * 100", f"Price move = ({fp(x)} - {fp(e)}) / {fp(e)} = {pct((x-e)/e*100, 3)}. ROE = {'+' if sgn > 0 else '-'}move x {lev} = {pct(ans)}."


@cat("funding_payment", "usd", 0.01)
def g_fund_pay(rng):
    n = rng.choice([1000, 2500, 5000, 10000, 20000, 50000, 120000]); r = round(rng.uniform(0.001, 0.05), 4); h = rng.choice([8, 24, 72, 168])
    per_h = rng.choice([1, 8]); periods = h // per_h; ans = n * r / 100 * periods
    qs = rng.choice([f"I hold a ${n:,} perp position. Funding is {fp(r)}% every {per_h} hour{'s' if per_h > 1 else ''} and I'm on the paying side. How much funding do I pay over {h} hours?{tail(rng)}",
                     f"Funding {fp(r)}% per {per_h}h on ${n} notional. Cost over {h} hours?"])
    return qs, ans, f"{n} * {fp(r)} / 100 * {periods}", f"{h} hours = {periods} funding periods. Cost = {money(n)} x {fp(r)}% x {periods} = {smart_money(ans)}."


@cat("apr_to_apy", "percent", 0.05)
def g_apr2apy(rng):
    apr = rng.choice([3, 4.5, 5, 6, 8, 10, 12, 15, 20, 25, 40, 60, 100]); n = rng.choice([("daily", 365), ("weekly", 52), ("monthly", 12)])
    ans = ((1 + apr / 100 / n[1]) ** n[1] - 1) * 100
    qs = rng.choice([f"A pool pays {fp(apr)}% APR, compounded {n[0]}. What's the APY?{tail(rng)}",
                     f"Convert {fp(apr)}% APR with {n[0]} compounding to APY."])
    return qs, ans, f"((1 + {fp(apr)}/100/{n[1]}) ** {n[1]} - 1) * 100", f"APY = (1 + APR/n)^n - 1 with n = {n[1]}: (1 + {fp(apr/100)}/{n[1]})^{n[1]} - 1 = {pct(ans)}."


@cat("apy_to_apr", "percent", 0.05)
def g_apy2apr(rng):
    apy = rng.choice([4, 5, 7, 8, 10, 12, 18, 25, 35, 50, 80]); n = 365; ans = ((1 + apy / 100) ** (1 / n) - 1) * n * 100
    qs = f"A vault shows {apy}% APY with daily compounding. What's the equivalent APR?{tail(rng)}"
    return qs, ans, f"((1 + {apy}/100) ** (1/365) - 1) * 365 * 100", f"APR = n x ((1 + APY)^(1/n) - 1) with n = 365: 365 x ((1 + {fp(apy/100)})^(1/365) - 1) = {pct(ans)}."


@cat("r_multiple", "r", 0.01)
def g_r(rng):
    e = P(rng); sp = rng.choice([1, 2, 3, 4, 5]) / 100; s = round(e * (1 - sp), 2); m = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, -1, -0.5])
    x = round(e + m * (e - s), 2); ans = (x - e) / (e - s)
    qs = rng.choice([f"Long entry {fp(e)}, stop {fp(s)}, I exited at {fp(x)}. How many R did I make?{tail(rng)}",
                     f"Entry ${fp(e)}, stop ${fp(s)}, exit ${fp(x)}. Result in R multiples?"])
    return qs, ans, f"({fp(x)} - {fp(e)}) / ({fp(e)} - {fp(s)})", f"1R = entry - stop = {num(e-s, 4)}. Gain = {fp(x)} - {fp(e)} = {num(x-e, 4)}. R = {num(x-e,4)} / {num(e-s,4)} = {num(ans, 2)}R."


@cat("rr_ratio", "ratio", 0.01)
def g_rr(rng):
    side = rng.choice(["long", "short"]); e = P(rng); d1 = rng.choice([1, 1.5, 2, 3, 4]) / 100; d2 = d1 * rng.choice([1, 1.5, 2, 2.5, 3, 4])
    if side == "long":
        s, t = round(e * (1 - d1), 2), round(e * (1 + d2), 2); ans = (t - e) / (e - s)
        ex = f"({fp(t)} - {fp(e)}) / ({fp(e)} - {fp(s)})"
    else:
        s, t = round(e * (1 + d1), 2), round(e * (1 - d2), 2); ans = (e - t) / (s - e)
        ex = f"({fp(e)} - {fp(t)}) / ({fp(s)} - {fp(e)})"
    qs = f"{side.title()} at {fp(e)}, stop {fp(s)}, target {fp(t)}. What's the reward to risk ratio?{tail(rng)}"
    return qs, ans, ex, f"Risk = {num(abs(e-s), 4)}, reward = {num(abs(t-e), 4)}. Reward:risk = {num(ans, 2)} to 1."


@cat("stop_from_risk", "usd", 0.01)
def g_stop_risk(rng):
    e = P(rng, ((5, 50), (50, 500), (500, 5000))); size = rng.choice([1, 2, 5, 10, 20, 50]); risk = rng.choice([25, 50, 100, 150, 200, 250, 500])
    if risk / size >= e * 0.5:
        size = size * 10
    ans = e - risk / size
    qs = f"I'm buying {size} coins at ${fp(e)} and want to lose at most ${risk} if stopped. Where should my stop be?{tail(rng)}"
    return qs, ans, f"{fp(e)} - {risk} / {size}", f"Allowed loss per coin = {risk} / {size} = {smart_money(risk/size)}. Stop = {fp(e)} - {num(risk/size, 4)} = {smart_money(ans)}."


@cat("avg_entry", "usd", 0.01)
def g_avg(rng):
    base = P(rng, ((5, 50), (50, 500), (500, 5000), (5000, 90000))); k = rng.choice([2, 3, 4]); usd = rng.choice([100, 200, 250, 500, 1000])
    ps = [round(base * rng.uniform(0.8, 1.2), 2) for _ in range(k)]
    units = sum(usd / p for p in ps); ans = usd * k / units
    qs = rng.choice([f"I bought ${usd} worth at each of these prices: {', '.join(fp(p) for p in ps)}. What's my average entry price?{tail(rng)}",
                     f"DCA buys of ${usd} at {', '.join(fp(p) for p in ps)}. Average cost per coin?"])
    ex = f"{usd * k} / (" + " + ".join(f"{usd}/{fp(p)}" for p in ps) + ")"
    return qs, ans, ex, f"Coins bought = {' + '.join(f'{usd}/{fp(p)}' for p in ps)} = {num(units, 6)}. Average = {money(usd*k)} / {num(units, 6)} = {smart_money(ans)}."


@cat("avg_down", "usd", 0.01)
def g_avg_down(rng):
    e1 = P(rng, ((5, 50), (50, 500), (500, 5000))); q1 = rng.choice([1, 2, 5, 10]); e2 = round(e1 * rng.uniform(0.7, 0.95), 2); q2 = rng.choice([1, 2, 5, 10, 20])
    ans = (e1 * q1 + e2 * q2) / (q1 + q2)
    qs = f"I hold {q1} at ${fp(e1)} and buy {q2} more at ${fp(e2)}. What's my new average entry (break-even price)?{tail(rng)}"
    return qs, ans, f"({fp(e1)} * {q1} + {fp(e2)} * {q2}) / ({q1} + {q2})", f"Cost = {fp(e1)} x {q1} + {fp(e2)} x {q2} = {money(e1*q1+e2*q2)}. Average = {money(e1*q1+e2*q2)} / {q1+q2} = {smart_money(ans)}."


@cat("basis_annual", "percent", 0.05)
def g_basis(rng):
    spot = P(rng, ((1000, 5000), (20000, 90000))); d = rng.choice([30, 45, 60, 90, 120, 180]); prem = rng.uniform(0.3, 4) / 100
    fut = float(round(spot * (1 + prem))); ans = (fut / spot - 1) * 365 / d * 100
    qs = f"Spot is {fp(spot)} and the {d}-day future is {fp(fut)}. What's the annualised basis (simple)?{tail(rng)}"
    return qs, ans, f"({fp(fut)} / {fp(spot)} - 1) * 365 / {d} * 100", f"Premium = {fp(fut)}/{fp(spot)} - 1 = {pct((fut/spot-1)*100, 3)}. Annualised = premium x 365/{d} = {pct(ans)}."


@cat("pred_market_ev", "usd", 0.01)
def g_pm(rng):
    price = rng.randint(5, 95) / 100; mine = min(0.99, max(0.01, price + rng.choice([-0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2]))); stake = rng.choice([10, 50, 100, 250, 500, 1000])
    shares = stake / price; ans = mine * shares - stake
    qs = rng.choice([f"A prediction market YES share costs ${fp(price)} and pays $1 if it happens. I think the real chance is {round(mine*100)}%. What's the expected profit on a ${stake} bet?{tail(rng)}",
                     f"YES trades at {round(price*100)} cents. My estimate is {round(mine*100)}%. Expected value of putting ${stake} on YES?"])
    mine = round(mine * 100) / 100; ans = mine * stake / price - stake
    return qs, ans, f"{fp(mine)} * {stake} / {fp(price)} - {stake}", f"Shares = {stake} / {fp(price)} = {num(stake/price, 4)}. EV = {fp(mine)} x {num(stake/price, 4)} x $1 - {stake} = {smart_money(ans)}."


@cat("pred_implied", "percent", 0.05)
def g_pm2(rng):
    yes = rng.randint(3, 97); no = 100 - yes + rng.choice([0, 1, 2, 3])
    ans = yes / (yes + no) * 100
    qs = f"On a prediction market YES is {yes} cents and NO is {no} cents. What's the implied probability of YES after removing the overround?{tail(rng)}"
    return qs, ans, f"{yes} / ({yes} + {no}) * 100", f"Total = {yes} + {no} = {yes+no} cents. Implied YES = {yes}/{yes+no} = {pct(ans)}."


@cat("depeg_loss", "usd", 0.01)
def g_depeg(rng):
    amt = rng.choice([1000, 2500, 5000, 10000, 25000, 50000]); p = round(rng.uniform(0.85, 0.995), 3)
    ans = amt * (1 - p)
    qs = f"I hold {amt:,} of a stablecoin that depegged to ${fp(p)}. How much value have I lost in dollars?{tail(rng)}"
    return qs, ans, f"{amt} * (1 - {fp(p)})", f"Loss = {amt:,} x (1 - {fp(p)}) = {smart_money(ans)}."


@cat("lp_fees", "usd", 0.01)
def g_lpfee(rng):
    vol = rng.choice([1, 2, 5, 10, 25, 50]) * 1e6; tier = rng.choice([0.01, 0.05, 0.3, 1]); share = rng.choice([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2])
    ans = vol * tier / 100 * share / 100
    qs = f"A pool does ${vol:,.0f} volume a day with a {fp(tier)}% fee tier. I own {fp(share)}% of the liquidity. My fees per day?{tail(rng)}"
    return qs, ans, f"{fp(vol)} * {fp(tier)} / 100 * {fp(share)} / 100", f"Pool fees = {money(vol, 0)} x {fp(tier)}% = {money(vol*tier/100)}. My share {fp(share)}% = {smart_money(ans)}."


@cat("gas_cost", "usd", 0.01)
def g_gas(rng):
    gas = rng.choice([21000, 46000, 65000, 120000, 180000, 250000]); gwei = rng.choice([0.01, 0.05, 0.5, 1, 3, 8, 15, 30, 60]); eth = P(rng, ((1800, 4800),))
    ans = gas * gwei * 1e-9 * eth
    qs = rng.choice([f"A transaction uses {gas:,} gas at {fp(gwei)} gwei. ETH is ${fp(eth)}. How much does it cost in dollars?{tail(rng)}",
                     f"Gas cost: {gas} gas, {fp(gwei)} gwei, ETH at {fp(eth)}. In USD?"])
    return qs, ans, f"{gas} * {fp(gwei)} * 1e-9 * {fp(eth)}", f"ETH used = {gas:,} x {fp(gwei)} gwei = {num(gas*gwei*1e-9, 8)} ETH. At {money(eth)} that's {smart_money(ans)}."


@cat("health_factor", "ratio", 0.01)
def g_hf(rng):
    coll = rng.choice([2000, 5000, 10000, 25000, 50000]); lt = rng.choice([0.78, 0.8, 0.825, 0.83, 0.86]); debt = round(coll * rng.uniform(0.2, 0.75), -1)
    ans = coll * lt / debt
    qs = f"On Aave I have ${coll:,} of collateral with a liquidation threshold of {fp(lt*100)}% and ${debt:,.0f} borrowed. What's my health factor?{tail(rng)}"
    return qs, ans, f"{coll} * {fp(lt)} / {fp(debt)}", f"Health factor = collateral x liquidation threshold / debt = {coll:,} x {fp(lt)} / {debt:,.0f} = {num(ans, 2)}{' (below 1 means liquidation)' if ans < 1 else ''}."


@cat("aave_liq_price", "usd", 0.01)
def g_aave_liq(rng):
    q = rng.choice([1, 2, 3, 5, 10]); eth = P(rng, ((1800, 4800),)); lt = rng.choice([0.8, 0.825, 0.83]); debt = round(q * eth * rng.uniform(0.3, 0.7), -1)
    ans = debt / (q * lt)
    qs = f"I supplied {q} ETH on Aave (liquidation threshold {fp(lt*100)}%) and borrowed ${debt:,.0f} USDC. At what ETH price does my health factor hit 1?{tail(rng)}"
    return qs, ans, f"{fp(debt)} / ({q} * {fp(lt)})", f"HF = 1 when {q} x price x {fp(lt)} = {debt:,.0f}. Price = {debt:,.0f} / ({q} x {fp(lt)}) = {smart_money(ans)}."


@cat("borrow_capacity", "usd", 0.01)
def g_borrow(rng):
    coll = rng.choice([1000, 3000, 5000, 12000, 30000]); ltv = rng.choice([0.7, 0.75, 0.8]); used = round(coll * rng.uniform(0, 0.5), -1)
    ans = coll * ltv - used
    qs = f"My collateral is ${coll:,} with a max LTV of {fp(ltv*100)}%. I've already borrowed ${used:,.0f}. How much more can I borrow?{tail(rng)}"
    return qs, ans, f"{coll} * {fp(ltv)} - {fp(used)}", f"Max borrow = {coll:,} x {fp(ltv)} = {money(coll*ltv)}. Minus {money(used)} already borrowed = {smart_money(ans)}."


@cat("compound_growth", "usd", 0.01)
def g_comp(rng):
    pv = rng.choice([500, 1000, 2500, 5000, 10000, 20000]); apy = rng.choice([3, 4, 5, 6, 8, 10, 12, 15]); yrs = rng.choice([1, 2, 3, 5, 10])
    ans = pv * (1 + apy / 100) ** yrs
    qs = rng.choice([f"If I put ${pv:,} into a vault at {apy}% APY, how much will I have after {yrs} year{'s' if yrs > 1 else ''}?{tail(rng)}",
                     f"${pv} compounding at {apy}% a year for {yrs} years. End value?"])
    return qs, ans, f"{pv} * (1 + {apy}/100) ** {yrs}", f"Value = {money(pv)} x (1 + {fp(apy/100)})^{yrs} = {smart_money(ans)}."


@cat("slippage_pct", "percent", 0.05)
def g_slip(rng):
    exp_ = P(rng, ((0.5, 5), (5, 50), (50, 500), (500, 5000))); s = rng.uniform(0.05, 3) / 100; got = round(exp_ * (1 + s), 4 if exp_ < 5 else 2)
    ans = (got - exp_) / exp_ * 100
    qs = f"I expected to buy at {fp(exp_)} but my average fill was {fp(got)}. What was my slippage in percent?{tail(rng)}"
    return qs, ans, f"({fp(got)} - {fp(exp_)}) / {fp(exp_)} * 100", f"Slippage = ({fp(got)} - {fp(exp_)}) / {fp(exp_)} = {pct(ans, 3)}."


@cat("tp_price", "usd", 0.01)
def g_tp(rng):
    side = rng.choice(["long", "short"]); e = P(rng); sd = rng.choice([1, 2, 2.5, 3, 5]) / 100; R = rng.choice([1.5, 2, 2.5, 3, 4])
    if side == "long":
        s = round(e * (1 - sd), 2); ans = e + R * (e - s); ex = f"{fp(e)} + {fp(R)} * ({fp(e)} - {fp(s)})"
    else:
        s = round(e * (1 + sd), 2); ans = e - R * (s - e); ex = f"{fp(e)} - {fp(R)} * ({fp(s)} - {fp(e)})"
    qs = f"{side.title()} entry {fp(e)}, stop {fp(s)}. Where do I put the take-profit for {fp(R)}R?{tail(rng)}"
    return qs, ans, ex, f"1R = {num(abs(e-s), 4)}. Target = {fp(e)} {'+' if side=='long' else '-'} {fp(R)} x {num(abs(e-s),4)} = {smart_money(ans)}."


@cat("margin_required", "usd", 0.01)
def g_margin(rng):
    n = rng.choice([500, 1000, 2500, 5000, 7500, 12000, 40000]); lev = rng.choice([2, 3, 5, 8, 10, 20, 25, 40]); fee = 0.045
    ans = n / lev
    qs = rng.choice([f"How much margin do I need for a ${n:,} position at {lev}x?{tail(rng)}",
                     f"I want {money(n)} notional with {lev}x leverage. Margin required?"])
    return qs, ans, f"{n} / {lev}", f"Margin = notional / leverage = {n:,} / {lev} = {smart_money(ans)}."


@cat("effective_leverage", "x", 0.01)
def g_efflev(rng):
    eq = rng.choice([500, 1000, 2000, 5000, 10000]); pos = [rng.choice([500, 1000, 2500, 4000, 8000, 15000]) for _ in range(rng.choice([1, 2, 3]))]
    ans = sum(pos) / eq
    qs = f"My account equity is ${eq:,} and I have open positions of {', '.join('$'+f'{p:,}' for p in pos)} notional. What's my effective leverage?{tail(rng)}"
    return qs, ans, f"({' + '.join(str(p) for p in pos)}) / {eq}", f"Total notional = {money(sum(pos), 0)}. Leverage = {money(sum(pos), 0)} / {money(eq, 0)} = {num(ans, 2)}x."


@cat("mcap_price", "usd", 0.0001)
def g_mcap(rng):
    sup = rng.choice([1e6, 1e8, 1e9, 2.1e7, 4.2e8, 1e10]); mcap = rng.choice([5e6, 2e7, 1e8, 5e8, 2e9, 1e10])
    ans = mcap / sup
    qs = f"A token has {sup:,.0f} tokens in circulation. What price gives it a ${mcap:,.0f} market cap?{tail(rng)}"
    return qs, ans, f"{fp(mcap)} / {fp(sup)}", f"Price = market cap / supply = {mcap:,.0f} / {sup:,.0f} = {smart_money(ans)}."


@cat("drawdown_recover", "percent", 0.05)
def g_dd(rng):
    d = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80, 90]); ans = (1 / (1 - d / 100) - 1) * 100
    qs = rng.choice([f"My portfolio is down {d}%. What percent gain do I need to get back to even?{tail(rng)}",
                     f"After a {d}% drawdown, how much do I need to make to recover?"])
    return qs, ans, f"(1 / (1 - {d}/100) - 1) * 100", f"Recovery = 1/(1 - {d/100:g}) - 1 = {pct(ans)}."


# ---------------------------------------------------------------- driver
def fmt_final(ans, unit, cat_name=""):
    if unit == "usd":
        return smart_money(ans)
    if unit == "usdc":
        return num(ans, 2) + " USDC"
    if unit == "percent":
        return pct(ans, 2)
    if unit == "bps":
        return num(ans, 2) + " bps"
    if unit == "r":
        return num(ans, 2) + "R"
    if unit == "ratio":
        return num(ans, 2)
    if unit == "x":
        return num(ans, 2) + "x"
    return num(ans, 4)


def make(n_per_cat, seed, cats=None):
    rng = random.Random(seed)
    out, seen = [], set()
    for name in (cats or list(C)):
        f, unit, tol = C[name]
        got = tries = 0
        while got < n_per_cat and tries < n_per_cat * 60:
            tries += 1
            q, ans, expr, steps = f(rng)
            if q in seen:
                continue
            v = calc(expr)
            assert abs(v - ans) <= 1e-6 * max(1, abs(ans)), (name, q, expr, v, ans)
            seen.add(q)
            out.append({"category": name, "question": q, "answer": round(ans, 8), "tol": tol, "unit": unit,
                        "expr": expr, "steps": steps, "final": fmt_final(ans, unit)})
            got += 1
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    import sys, json
    rows = make(3, 5)
    for r in rows[:12]:
        print(json.dumps(r, ensure_ascii=False))
    print(len(C), "categories;", len(rows), "rows")
