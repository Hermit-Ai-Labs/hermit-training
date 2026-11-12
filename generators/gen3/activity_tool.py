#!/usr/bin/env python3
"""activity_tool: get_activity for direct activity requests, and faithful reading of the rows.

Writes (one run, both files):
  data3/activity_tool.jsonl   train rows, --seed (default 3), --n (default 250)
  eval3/activity_tool.jsonl   held-out eval items, --eval-seed (default 9313), --eval-n (default 30)

Record shape (train):  {"messages": [...], "tools": true, "meta": {"source": "activity_tool", ...}}
Eval item shape:       {id, kind, tools, bio, context, reference, expect, world, checks, answer_checks, meta}
Held-out means: eval uses only templates whose index % 5 == 0 and only the eval slot values (symbols, limits), and no
user text (normalised) that occurs in train or in eval2.

Nothing is LLM-written. Every answer is a Python f-string filled from the tool result; every count (items, in progress,
failed, shown) is computed with len()/sum() in code. Tool results follow spec/hermit_tools_v2.meta.json.
gen2 helpers are imported (hermit_fmt for system-prompt user suffixes, addresses, tool-call format; sim for prices and
balances), never copied or edited.

Run:  python gen3/activity_tool.py
"""
import argparse
import json
import os
import random
import re
import string
import sys
from collections import Counter
from decimal import Decimal

sys.dont_write_bytecode = True          # never write .pyc files next to gen2
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(TRAIN_DIR, "gen2"))
import hermit_fmt as H      # noqa: E402  (gen2)
import sim                  # noqa: E402  (gen2)

SPEC = os.path.join(TRAIN_DIR, "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
SYSTEM_V2 = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().rstrip("\n")
assert "{BIO}" in SYSTEM_V2
# render with the 19-tool block (runtime only, gen2 files untouched)
H.TOOLS = TOOLS_V2
H.TOOL_NAMES = [t["function"]["name"] for t in TOOLS_V2]
H.TOOL_SCHEMA = TOOL_SCHEMA
H.TOOLS_BLOCK = H.tools_block()

NOTE = "Reviews the user has not approved yet do not appear here."
SRC = "activity_tool"

# ------------------------------------------------------------------ small helpers
def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def fmt_dec(x):
    """Shortest plain decimal text of a number (no exponent, no trailing zeros)."""
    d = Decimal(repr(float(x))) if not isinstance(x, Decimal) else x
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def sig3(x):
    return fmt_dec(Decimal(f"{x:.3g}"))


def plural(n, one, many):
    return one if n == 1 else many


CHAIN_NAME = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "solana": "Solana", "polygon": "Polygon",
              "bnb": "BNB Chain", "bsc": "BNB Chain", "monad": "Monad"}
ROW_CHAINS = ["Ethereum", "Base", "Arbitrum", "Solana", "Polygon", "BNB Chain"]
AAVE_CHAINS = ["Ethereum", "Base", "Arbitrum"]
STATUS_TXT = {"confirmed": "confirmed", "failed": "failed", "in_progress": "in progress"}

# ------------------------------------------------------------------ activity rows
STABLE = ("USDC", "USDT", "DAI")
SEND_USD = [5, 10, 15, 20, 25, 40, 50, 75, 100, 150, 200, 250, 300, 500, 1000]
STABLE_AMTS = ["0.5", "1", "2", "5", "10", "12.5", "20", "25", "50", "75", "100", "120", "150", "250", "500", "1000"]
RAND_SYMS = ["USDC", "USDC", "USDC", "USDT", "ETH", "SOL", "BNB", "WIF", "BONK", "HYPE", "DOGE", "LINK", "AVAX", "ARB", "UNI",
             "JUP", "SUI", "DAI", "AAVE"]
RAND_KINDS = ["send", "send", "swap", "bridge", "buy", "buy", "deposit", "withdraw"]


def short_addr(rng):
    r = rng.random()
    if r < 0.12:
        a = H.sol_addr(rng)
        return a[:4] + "…" + a[-4:]
    a = H.evm_addr(rng)
    if r < 0.30:
        return a[:6] + "…" + a[-4:]
    return a[:4] + "…" + a[-2:]


def amount_for(rng, w, sym):
    if sym in STABLE:
        return rng.choice(STABLE_AMTS)
    price = w.prices.get(sym, 1.0)
    return sig3(rng.choice(SEND_USD) / price)


def make_row(rng, w, kind, sym=None):
    """Return (what, meta). meta = {kind, syms, amts} is used for matching only, never shown to the model."""
    if kind == "send":
        sym = sym or rng.choice(RAND_SYMS)
        amt = amount_for(rng, w, sym)
        if rng.random() < 0.08:
            to = rng.choice(["alice", "mum", "noah", "vault", "dev", "maya", "sam"]) + ".eth"
        else:
            to = short_addr(rng)
        return f"Sent {amt} {sym} to {to}", {"kind": "send", "syms": [sym], "amts": [amt]}
    if kind == "swap":
        a = sym or rng.choice(["USDC", "USDC", "USDT", "ETH", "SOL", "BNB"])
        b = rng.choice([s for s in RAND_SYMS if s != a])
        amt = amount_for(rng, w, a)
        return (f"Swap {amt} {a} to {b} on {rng.choice(ROW_CHAINS)}",
                {"kind": "swap", "syms": [a, b], "amts": [amt]})
    if kind == "bridge":
        a = sym or rng.choice(["USDC", "USDC", "USDT", "ETH"])
        b = a if rng.random() < 0.85 else rng.choice([s for s in ("USDC", "ETH", "USDT") if s != a])
        c1, c2 = rng.sample(ROW_CHAINS, 2)
        amt = amount_for(rng, w, a)
        return (f"Swap {amt} {a} on {c1} to {b} on {c2}",
                {"kind": "bridge", "syms": sorted({a, b}), "amts": [amt]})
    if kind == "buy":
        sym = sym or rng.choice(["WIF", "BONK", "ETH", "SOL", "HYPE", "DOGE", "BNB", "JUP", "LINK", "SUI"])
        usd = str(rng.choice([5, 10, 10, 15, 20, 25, 50, 100]))
        q = sig3(Decimal(usd) / Decimal(str(w.prices.get(sym, 1.0))))
        return (f"Bought {q} {sym} with {usd} USDC",
                {"kind": "buy", "syms": [sym, "USDC"], "amts": [q, usd]})
    if kind in ("deposit", "withdraw"):
        sym = sym or rng.choice(["USDC", "USDC", "USDT", "ETH", "DAI"])
        amt = amount_for(rng, w, sym)
        ch = rng.choice(AAVE_CHAINS)
        if kind == "deposit":
            what = f"Deposited {amt} {sym} into Aave v3 on {ch}"
        else:
            what = f"Withdrew {amt} {sym} from Aave v3 on {ch}"
        return what, {"kind": kind, "syms": [sym], "amts": [amt]}
    raise ValueError(kind)


KIND_CLASS = {"swap": {"swap", "bridge"}, "bridge": {"bridge"}, "send": {"send"}, "buy": {"buy"},
              "deposit": {"deposit"}, "withdraw": {"withdraw"}}


def match(metas, ukind, sym=None, amt=None):
    return [i for i, m in enumerate(metas)
            if m["kind"] in KIND_CLASS[ukind] and (sym is None or sym in m["syms"]) and (amt is None or amt in m["amts"])]


NOW_DAY = [3, 4, 5, 6, 7, 8, 9, 10]


def gen_times(rng, n, today):
    """n times, newest first, as 'YYYY-MM-DD HH:MM'. today=True keeps all of them on one calendar day."""
    if n == 0:
        return []
    day = rng.choice(NOW_DAY)
    mins = rng.randint(15 * 60, 23 * 60 + 50) if today else rng.randint(8 * 60, 23 * 60 + 50)
    t = day * 1440 + mins
    out = [t]
    for _ in range(n - 1):
        if today:
            gap = rng.randint(2, 45)
        else:
            gap = rng.randint(2, 90) if rng.random() < 0.7 else rng.randint(180, 3000)
        t -= gap
        out.append(t)
    res = []
    for t in out:
        d, m = divmod(t, 1440)
        if d < 1:
            d, m = 1, 0
        res.append(f"2026-11-{d:02d} {m // 60:02d}:{m % 60:02d}")
    return res


def assign_status(rng, times, force_pending=False):
    """Statuses, in_progress only for the newest 4 rows and only when close to the newest one."""
    out = []
    for i, t in enumerate(times):
        age_ok = i < 4 and (i == 0 or (_mins(times[0]) - _mins(t)) <= 150)
        r = rng.random()
        if r < 0.20 and age_ok:
            out.append("in_progress")
        elif r < 0.38:
            out.append("failed")
        else:
            out.append("confirmed")
    if force_pending and times and "in_progress" not in out:
        out[rng.randrange(min(len(times), 3))] = "in_progress"
    return out


def _mins(t):
    d = int(t[8:10])
    return d * 1440 + int(t[11:13]) * 60 + int(t[14:16])


def random_metas(rng, w, n):
    rows = []
    for _ in range(n):
        rows.append(make_row(rng, w, rng.choice(RAND_KINDS)))
    return rows


def build_activity(rows, times, statuses):
    return [{"status": s, "time": t, "what": what} for (what, _m), t, s in zip(rows, times, statuses)]


# ------------------------------------------------------------------ simulated tools (also the replay used by the self-check)
def t_get_activity(world, limit=None):
    if not world.get("has_wallet", True):
        return {"error": f"No wallet on this {world['device']} yet"}
    rows = world["activity"]
    cap = 10 if limit is None else int(limit)
    rows = rows[:cap]
    out = [{"status": r["status"], "time": r["time"], "what": r["what"]} for r in rows]
    p = sum(1 for r in out if r["status"] == "in_progress")
    note = NOTE if out else "No activity yet. " + NOTE
    return {"activity": out, "in_progress": p, "note": note}


def t_get_balances(world, chain=None):
    hold = world["holdings"]
    if chain:
        hold = [h for h in hold if h["chain"] == chain]
    return {"holdings": hold}


TOOL_FN = {"get_activity": t_get_activity, "get_balances": t_get_balances}


def call_tool(world, name, args):
    return TOOL_FN[name](world, **args)


# ------------------------------------------------------------------ user phrasing
def split_pool(lst, split):
    """Held-out rule: index % 5 == 0 is eval-only."""
    return [t for i, t in enumerate(lst) if (i % 5 == 0) == (split == "eval")]


GENERAL = [
    "show my recent activity", "show me my recent activity", "recent activity", "my recent activity",
    "what's my recent activity", "activity", "my activity", "check my activity", "pull up my activity",
    "show activity", "show me what I did today", "what did I do today", "what have I done today",
    "what did I do in the wallet today", "anything I did today", "show my transactions", "my transactions",
    "recent transactions", "show me my recent transactions", "any recent transactions", "what happened in my wallet",
    "what's been happening in my wallet", "what happened in my wallet today", "history", "my history", "show history",
    "show my history", "transaction history", "show me my transaction history", "tx history", "show my tx history",
    "wallet history", "what have I sent or swapped lately", "what moves have I made lately",
    "list my latest transactions", "latest activity", "what's the latest on my wallet", "anything new in my activity",
    "can you show me what I did recently", "what's in my activity feed", "give me my activity",
    "let me see my recent transactions", "what did I do recently", "what did I do lately",
    "show me everything I did recently", "tell me what I've done in the app", "what's my wallet been up to",
    "rundown of my recent transactions", "summarize my recent activity", "go through my activity for me",
    "what have I done this week", "what did I just do", "what were my latest moves", "pull my transaction list",
    "show my wallet activity", "do I have any recent transactions", "did anything happen on my wallet",
    "what's my transaction log", "recent txs", "show my recent txs", "activity log", "what's in my wallet history",
]
LIMIT = [
    "last {n} transactions", "show my last {n} transactions", "last {n} activity items", "show me the last {n}",
    "my last {n} transactions", "last {n} things I did", "{n} most recent transactions",
    "show the {n} latest transactions", "latest {n} moves", "what were my last {n} transactions",
    "give me my last {n} txs", "recent activity, just the last {n}", "only the last {n} please",
    "top {n} recent transactions", "show last {n} activity", "my last {n} moves", "last {n} txs",
    "can you show the last {n} things in my activity", "pull up my {n} latest transactions",
    "I only want to see the last {n} transactions", "list the last {n} items in my history",
    "{n} latest activity items", "show me {n} recent transactions", "what are my last {n} actions",
    "last {n} things in my wallet", "history, last {n} only", "get my last {n} transactions", "the last {n} I did",
    "last {n} wallet transactions", "show my {n} most recent txs",
]
PENDING = [
    "what's pending", "what's pending?", "anything pending", "any pending transactions", "what's still pending",
    "anything in progress", "what's in progress", "anything still processing", "is anything still processing",
    "what's processing right now", "any transactions in flight", "is anything still going",
    "do I have anything pending", "do I have anything in progress", "any pending items", "anything waiting to confirm",
    "what's not confirmed yet", "anything not confirmed yet", "what hasn't confirmed yet", "show me pending transactions",
    "pending transactions", "show pending", "list what's in progress", "is anything unfinished", "anything still running",
    "any transactions still going through", "check for pending stuff", "are any of my transactions still pending",
    "what's in flight",
]
KIND_NOUNS = {  # kind -> user plural words
    "swap": ["swaps", "swap transactions"], "bridge": ["bridges", "bridge transfers"],
    "send": ["sends", "transfers", "payments"], "buy": ["buys", "purchases"],
    "deposit": ["deposits", "Aave deposits"], "withdraw": ["withdrawals", "Aave withdrawals", "withdraws"],
}
ANSWER_PLURAL = {"swap": "swaps", "bridge": "bridges", "send": "sends", "buy": "buys", "deposit": "deposits",
                 "withdraw": "withdrawals"}
ANSWER_SING = {"swap": "swap", "bridge": "bridge", "send": "send", "buy": "buy", "deposit": "deposit",
               "withdraw": "withdrawal"}
KIND_PENDING = [
    "any {kp} in progress", "any {kp} in progress?", "are any {kp} still pending", "any {kp} pending",
    "got any {kp} still processing", "are there any {kp} in progress", "any {kp} still in progress",
    "do I have any {kp} pending", "any {kp} that haven't finished", "check if any {kp} are still pending",
    "any {kp} stuck", "show me my pending {kp}", "are my {kp} still processing", "are any of my {kp} unfinished",
    "still any {kp} going", "anything pending on {kp}", "any {kp} waiting to confirm", "is there a {kp1} still in progress",
]
LATEST = [
    "my last transaction", "what was my last transaction", "latest transaction", "my latest transaction",
    "what was the last thing I did", "last thing I did in the wallet", "show my most recent transaction",
    "most recent transaction", "what was my last move", "what's my newest transaction", "newest activity item",
    "the last thing in my activity", "what did I do last", "my last tx",
]
KIND_SING = {"send": ["send", "transfer", "payment"], "swap": ["swap", "exchange"], "bridge": ["bridge"],
             "buy": ["buy", "purchase"], "deposit": ["deposit"], "withdraw": ["withdrawal", "withdraw"]}
LAST_KIND = [
    "my last {kw}", "what was my last {kw}", "show my last {kw}", "my latest {kw}", "latest {kw}",
    "did my last {kw} confirm", "is my last {kw} confirmed", "how's my last {kw}", "what happened with my last {kw}",
    "status of my last {kw}", "the last {kw} I made", "check my last {kw}", "most recent {kw}", "my most recent {kw}",
    "what's the status of my latest {kw}",
]
STATUS = [
    "did {thing} go through", "did {thing} go through?", "is {thing} done", "is {thing} done yet?",
    "has {thing} confirmed", "is {thing} confirmed", "did {thing} work", "what's up with {thing}",
    "any update on {thing}", "what's the status of {thing}", "status of {thing}", "{thing} status?",
    "did {thing} land", "did {thing} arrive", "has {thing} finished", "is {thing} complete",
    "is {thing} still pending", "is {thing} still in progress", "did {thing} fail", "did {thing} actually happen",
    "can you check {thing}", "check {thing} for me", "how's {thing} doing", "where's {thing} at",
    "has {thing} cleared", "see if {thing} is done", "look up {thing}", "what happened to {thing}",
    "is {thing} stuck", "did {thing} finish", "has {thing} gone through", "did {thing} get through",
    "{thing}, is it done", "{thing}: done?", "any news on {thing}", "is {thing} through yet",
    "tell me about {thing}", "how did {thing} go", "can you tell me if {thing} worked", "did {thing} complete",
]
THINGS = {
    ("send", "symamt"): ["my {amt} {sym} send", "the {amt} {sym} transfer", "the {amt} {sym} I sent", "my {amt} {sym} payment",
                         "the {amt} {sym} send", "that {amt} {sym} transfer", "the {sym} transfer of {amt}",
                         "my send of {amt} {sym}", "the transaction sending {amt} {sym}", "the {amt} {sym} I just sent"],
    ("send", "sym"): ["my {sym} send", "the {sym} transfer", "the {sym} I sent", "my {sym} payment", "the {sym} send",
                      "that {sym} transfer", "my {sym} transfer", "the {sym} I just sent", "the {sym} transaction I sent",
                      "the {sym} payment I made"],
    ("send", "none"): ["my send", "the transfer", "the send", "my transfer", "the one I sent", "that payment",
                       "the money I sent", "the transfer I made", "the send I just made", "my payment"],
    ("swap", "symamt"): ["my {amt} {sym} swap", "the {amt} {sym} swap", "the swap of {amt} {sym}", "the {amt} {sym} swap I made",
                         "that {amt} {sym} swap", "the {sym} swap for {amt}", "my {amt} {sym} exchange",
                         "the swap of {amt} {sym} I started", "that swap of {amt} {sym}", "the {amt} {sym} one I swapped"],
    ("swap", "sym"): ["my {sym} swap", "the {sym} swap", "the swap into {sym}", "the swap out of {sym}",
                      "the {sym} swap I made", "that {sym} swap", "my swap with {sym}", "the swap involving {sym}",
                      "the {sym} exchange", "the {sym} swap I just did"],
    ("swap", "none"): ["my swap", "the swap", "that swap", "the swap I made", "my exchange", "the swap I just made",
                       "the swap I started", "that exchange", "the swap I did", "the swap I kicked off"],
    ("bridge", "symamt"): ["my {amt} {sym} bridge", "the {amt} {sym} bridge", "the bridge of {amt} {sym}",
                           "the {amt} {sym} cross-chain transfer", "that {amt} {sym} bridge", "the {sym} bridge for {amt}",
                           "my {amt} {sym} cross-chain move", "the bridge with {amt} {sym}", "the {amt} {sym} I bridged"],
    ("bridge", "sym"): ["my {sym} bridge", "the {sym} bridge", "that {sym} bridge", "the {sym} cross-chain transfer",
                        "the {sym} I bridged", "my {sym} cross-chain move", "the {sym} bridge I made",
                        "the {sym} bridge I just did", "the bridge with {sym}", "the {sym} bridging"],
    ("bridge", "none"): ["my bridge", "the bridge", "that bridge", "the bridge I made", "the cross-chain transfer",
                         "my bridging", "the cross-chain move", "the bridge I just started", "the bridge I did",
                         "the bridge I kicked off"],
    ("buy", "symamt_usd"): ["the ${amt} {sym} buy", "my ${amt} {sym} buy", "the ${amt} {sym} purchase",
                            "the {amt} USDC {sym} buy", "my {amt} USDC buy of {sym}", "the {sym} buy for ${amt}",
                            "the {sym} buy with {amt} USDC", "that ${amt} {sym} buy", "my {amt} USDC {sym} purchase",
                            "the {sym} order for ${amt}"],
    ("buy", "symamt_tok"): ["the {amt} {sym} buy", "my {amt} {sym} purchase", "the buy of {amt} {sym}",
                            "that {amt} {sym} buy", "the {amt} {sym} I bought", "the purchase of {amt} {sym}",
                            "my buy of {amt} {sym}", "the {amt} {sym} order", "the {amt} {sym} purchase",
                            "the {amt} {sym} buy I made"],
    ("buy", "sym"): ["the {sym} buy", "my {sym} purchase", "the {sym} I bought", "my {sym} buy", "that {sym} purchase",
                     "the {sym} buy I made", "the {sym} order", "the {sym} I just bought", "my {sym} buy just now",
                     "the {sym} purchase I made"],
    ("buy", "none"): ["my buy", "the purchase", "my purchase", "the buy", "that buy", "the one I bought",
                      "the buy I made", "that purchase", "the purchase I made", "the buy I just did"],
    ("deposit", "symamt"): ["my {amt} {sym} deposit", "the {amt} {sym} deposit into Aave", "the {amt} {sym} Aave deposit",
                            "my deposit of {amt} {sym}", "the {amt} {sym} I put into Aave", "that {amt} {sym} deposit",
                            "the {amt} {sym} deposit I made", "my {amt} {sym} deposit on Aave", "the {sym} deposit of {amt}",
                            "the {amt} {sym} I deposited"],
    ("deposit", "sym"): ["my {sym} deposit", "the {sym} Aave deposit", "the {sym} deposit into Aave", "the {sym} I deposited",
                         "my {sym} deposit on Aave", "that {sym} deposit", "the {sym} I put in Aave",
                         "the {sym} deposit I made", "the {sym} Aave supply", "the {sym} deposit I just did"],
    ("deposit", "none"): ["my Aave deposit", "the deposit", "my deposit", "the deposit into Aave", "that deposit",
                          "the Aave deposit", "the one I put into Aave", "the deposit I made", "the deposit I just did",
                          "the Aave supply"],
    ("withdraw", "symamt"): ["my {amt} {sym} withdrawal", "the {amt} {sym} Aave withdrawal", "my withdrawal of {amt} {sym}",
                             "the {amt} {sym} withdraw", "my {amt} {sym} withdraw from Aave", "that {amt} {sym} withdrawal",
                             "the {amt} {sym} I withdrew", "the {sym} withdrawal of {amt}", "the {amt} {sym} I took out of Aave",
                             "my {amt} {sym} Aave withdraw"],
    ("withdraw", "sym"): ["my {sym} withdrawal", "the {sym} Aave withdrawal", "the {sym} withdrawal from Aave",
                          "the {sym} I withdrew", "my {sym} withdraw", "that {sym} withdrawal",
                          "the {sym} I took out of Aave", "the {sym} withdrawal I made", "my {sym} Aave withdraw",
                          "the {sym} withdrawal I just did"],
    ("withdraw", "none"): ["my Aave withdrawal", "the withdrawal", "my withdrawal", "the withdraw",
                           "that withdrawal", "the withdrawal from Aave", "the one I took out of Aave",
                           "the withdrawal I made", "the withdrawal I just did", "the Aave withdraw"],
}
BALANCE = [
    "what's my balance", "how much do I have", "what do I hold", "show my balances", "how much money do I have",
    "what's in my wallet", "wallet balance", "check my balance", "what do I own", "how much crypto do I have",
    "show me my holdings", "my holdings", "what coins do I have", "balance?", "what's my balance right now",
]
BALANCE_SYM = ["how much {sym} do I have", "do I have any {sym}", "what's my {sym} balance",
               "how much {sym} is in my wallet", "{sym} balance?", "how many {sym} do I hold"]
HOWTO = [
    "how do I see my history", "where can I see my past transactions", "where is the activity page",
    "how can I look at old transactions", "where do I find my transaction history", "how do I check my activity",
    "where's my history in the app", "how do I see what I've sent", "where do I look to see my swaps",
    "how do I view my wallet history", "where is transaction history in hermit", "how can I see my activity in the app",
]
HASH = [
    "what is a transaction hash", "what's a tx hash", "what does tx hash mean", "what is a txid",
    "what is a transaction id", "explain transaction hash", "what's a tx id", "what does a transaction hash do",
    "what is a tx hash used for", "can you explain what a transaction hash is", "tx hash meaning",
    "what is a hash in a transaction",
]
PREFIX = ["hey ", "hey hermit, ", "yo ", "hi, ", "ok ", "quick one: ", "hermit ", "so ", "pls ", "hmm "]
SUFFIX = [" pls", " please", " thanks", " thx", " real quick", " for me", " rn", " asap"]

TRAIN_SYMS = {"send": ["USDC", "USDT", "ETH", "SOL", "BNB", "WIF"], "swap": ["USDC", "ETH", "SOL", "BNB", "WIF", "HYPE"],
              "bridge": ["USDC", "USDT", "ETH"], "buy": ["WIF", "BONK", "ETH", "SOL", "BNB", "HYPE", "DOGE"],
              "deposit": ["USDC", "USDT", "ETH"], "withdraw": ["USDC", "USDT", "ETH"]}
EVAL_SYMS = {"send": ["LINK", "AVAX", "ARB", "DAI", "USDC", "UNI"], "swap": ["LINK", "AVAX", "ARB", "UNI", "JUP", "SUI"],
             "bridge": ["DAI", "USDC", "ETH"], "buy": ["JUP", "LINK", "AVAX", "SUI", "ARB", "UNI"],
             "deposit": ["DAI", "USDC", "ETH"], "withdraw": ["DAI", "USDC", "ETH"]}
TRAIN_LIMITS = [2, 3, 3, 4, 5, 5, 6, 8, 10]
EVAL_LIMITS = [7, 9, 11, 12, 15, 3, 5]
TRAIN_BAL_SYMS = ["ETH", "USDC", "SOL", "BNB", "USDT", "HYPE"]
EVAL_BAL_SYMS = ["LINK", "AVAX", "UNI", "ARB", "USDC", "ETH"]


def _typo(rng, word):
    op = rng.choice(["swap", "drop", "double", "swap"])
    i = rng.randrange(1, len(word) - 1)
    if op == "swap" and word[i] != word[i + 1]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        vs = [j for j in range(1, len(word) - 1) if word[j] in "aeiou"]
        if vs:
            j = rng.choice(vs)
            return word[:j] + word[j + 1:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    return word


def _typo_literal(rng, lit):
    words = list(re.finditer(r"[A-Za-z]{4,}", lit))
    if not words:
        return lit
    m = rng.choice(words)
    return lit[:m.start()] + _typo(rng, m.group(0)) + lit[m.end():]


def compose(rng, tpl, slots, allow_decor=True):
    """Fill {fields} (protected from typos), then apply prefix/suffix, casing, punctuation and typo layers."""
    pieces = []
    for lit, field, _spec, _conv in string.Formatter().parse(tpl):
        if lit:
            pieces.append([lit, False])
        if field is not None:
            pieces.append([str(slots[field]), True])
    if allow_decor:
        if rng.random() < 0.07:
            pieces.insert(0, [rng.choice(PREFIX), False])
        if rng.random() < 0.12:
            pieces.append([rng.choice(SUFFIX), False])
    r = rng.random()
    lits = [p for p in pieces if not p[1]]
    if lits:
        if r < 0.50:
            pass
        elif r < 0.78:
            if not pieces[0][1]:
                pieces[0][0] = pieces[0][0][:1].upper() + pieces[0][0][1:]
        elif r < 0.93:
            for p in lits:
                p[0] = p[0].lower()
        elif r < 0.98:
            if not pieces[-1][1]:
                pieces[-1][0] = pieces[-1][0].rstrip("?.!")
        else:
            for p in lits:
                p[0] = p[0].upper()
        if rng.random() < 0.15:
            k = rng.choice([1, 1, 2])
            for _ in range(k):
                cand = [p for p in pieces if not p[1] and re.search(r"[A-Za-z]{4,}", p[0])]
                if cand:
                    p = rng.choice(cand)
                    p[0] = _typo_literal(rng, p[0])
        if rng.random() < 0.06:
            for p in lits:
                p[0] = (p[0].replace("please", "pls").replace("Please", "Pls").replace("right now", "rn")
                        .replace(" rn", " rn"))
    return "".join(p[0] for p in pieces).strip()


# ------------------------------------------------------------------ conversation builder
class Conv:
    def __init__(self, rng, world, bio):
        self.rng = rng
        self.bio = bio
        sysmsg = SYSTEM_V2.replace("{BIO}", bio)
        self.msgs = [{"role": "system", "content": H.with_user(sysmsg, rng).replace("Face ID", bio)}]
        self.world = world
        self.call_worlds = []
        self.expect = []

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.expect = []

    def call(self, name, args):
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]})
        res = call_tool(self.world, name, args)
        self.msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
        self.call_worlds.append(json.loads(json.dumps(self.world)))
        self.expect.append({"calls": [{"name": name, "arguments": args}]})
        return res

    def say(self, text):
        self.msgs.append({"role": "assistant", "content": text})
        self.expect.append({"text": True})


# ------------------------------------------------------------------ answers (all numbers computed in code)
def line(r):
    return f"- {r['what']}, {STATUS_TXT[r['status']]}, {r['time']}"


def fail_tail(rng, shown, single=False):
    f = sum(1 for r in shown if r["status"] == "failed")
    if f == 0:
        return ""
    ask = rng.choice(["I can check your balances.", "I can check your balances if you want.",
                      "Want me to check your balances?", "I can check your balances for you."])
    if f == 1:
        lead = "It shows as failed." if single else rng.choice(["One shows as failed.", "One of them shows as failed."])
    else:
        lead = rng.choice([f"{f} show as failed.", f"{f} of them show as failed."])
    return f"{lead} {ask}"


def maybe_note(rng, p=0.1):
    return rng.choice([" Reviews you haven't approved yet don't show here.",
                       " Reviews you haven't approved yet don't appear here."]) if rng.random() < p else ""


def headline(rng, rows, p):
    n = len(rows)
    items = f"{n} recent {plural(n, 'item', 'items')}"
    if p > 0:
        opts = [f"{items}, {p} still in progress.",
                f"You have {items}, and {p} {plural(p, 'is', 'are')} still in progress.",
                f"Latest activity: {items}, {p} in progress."]
        if n > 1:
            opts.append(f"{p} of your {n} recent items {plural(p, 'is', 'are')} still in progress.")
    else:
        opts = [f"{items}, none in progress.", f"You have {items}, nothing in progress.",
                f"Latest activity: {items}, none in progress.", f"Nothing is in progress. {items} in total."]
    return rng.choice(opts)


def ans_empty(rng, res):
    return rng.choice(["No activity yet.", "No activity yet.", "No activity yet. Reviews you haven't approved yet don't show here.",
                       "There's no activity yet."])


def ans_error(rng, res):
    m = re.search(r"No wallet on (this \w+) yet", res["error"])
    dev = m.group(1)
    return rng.choice([f"You don't have a wallet on {dev} yet, so there's no activity to show.",
                       f"I can't show activity yet because there's no wallet on {dev}.",
                       f"There's no wallet on {dev} yet, so Activity is empty.",
                       f"I can't load your activity: no wallet on {dev} yet."])


def ans_list(rng, res, kind_label="recent"):
    rows = res["activity"]
    p = res["in_progress"]
    shown = rows[:5]
    head = headline(rng, rows, p)
    k = len(shown)
    if len(rows) > 5:
        lead = rng.choice([f"Here are the newest {k}:", f"The newest {k}:", f"Showing the latest {k}:"])
    else:
        lead = rng.choice(["", "", "Here they are:", "The list:"])
    body = "\n".join(line(r) for r in shown)
    out = head + ("\n" + lead if lead else "") + "\n" + body
    tail = fail_tail(rng, shown, single=(len(shown) == 1))
    extra = maybe_note(rng) if p > 0 else ""
    if tail:
        out += "\n\n" + tail
    if extra:
        out += "\n\n" + extra.strip()
    return out, shown


def ans_pending(rng, res):
    rows = res["activity"]
    pend = [r for r in rows if r["status"] == "in_progress"]
    if pend:
        head = rng.choice([f"{len(pend)} still in progress:", f"{len(pend)} {plural(len(pend), 'item is', 'items are')} still in progress:",
                           f"You have {len(pend)} in progress:"])
        return head + "\n" + "\n".join(line(r) for r in pend), pend
    r0 = rows[0]
    head = rng.choice(["Nothing is in progress right now. The latest item:", "Nothing pending. Your latest item:",
                       "No, nothing is in progress. The latest item:"])
    out = head + "\n" + line(r0)
    tail = fail_tail(rng, [r0], single=True)
    if tail:
        out += "\n\n" + tail
    return out, [r0]


def ans_kind_pending(rng, res, metas, ukind):
    rows = res["activity"]
    idx = [i for i in match(metas[:len(rows)], ukind) if rows[i]["status"] == "in_progress"]
    word = ANSWER_PLURAL[ukind]
    if idx:
        sel = [rows[i] for i in idx]
        head = rng.choice([f"{len(sel)} {plural(len(sel), ANSWER_SING[ukind], word)} still in progress:",
                           f"Yes, {len(sel)} still in progress:", f"{len(sel)} in progress:"])
        return head + "\n" + "\n".join(line(r) for r in sel), sel
    others = [r for r in rows if r["status"] == "in_progress"]
    head = rng.choice([f"No {word} in progress.", f"No {word} are in progress.", f"None of your {word} are in progress."])
    if others:
        out = (head + " " + f"{len(others)} other {plural(len(others), 'item is', 'items are')} still in progress:\n" +
               "\n".join(line(r) for r in others))
        return out, others
    return head, []


def ans_latest(rng, res):
    r0 = res["activity"][0]
    head = rng.choice(["Your latest item:", "Your most recent item:", "The latest in Activity:", "Latest item:"])
    out = head + "\n" + line(r0)
    tail = fail_tail(rng, [r0], single=True)
    if tail:
        out += "\n\n" + tail
    return out, [r0]


def ans_last_kind(rng, res, metas, ukind):
    rows = res["activity"]
    idx = match(metas[:len(rows)], ukind)
    if not idx:
        return "I don't see it in Activity." + maybe_note(rng, 0.4), []
    r = rows[idx[0]]
    head = rng.choice([f"Your last {ANSWER_SING[ukind]}:", f"Your latest {ANSWER_SING[ukind]} in Activity:",
                       f"The most recent {ANSWER_SING[ukind]}:"])
    out = head + "\n" + line(r)
    tail = fail_tail(rng, [r], single=True)
    if tail:
        out += "\n\n" + tail
    return out, [r]


def ans_specific(rng, res, metas, ukind, sym, amt):
    rows = res["activity"]
    idx = match(metas[:len(rows)], ukind, sym, amt)
    if not idx:
        return "I don't see it in Activity." + maybe_note(rng, 0.5), []
    r = rows[idx[0]]
    w, t = r["what"], r["time"]
    if r["status"] == "confirmed":
        out = rng.choice([f"Activity shows it as confirmed: {w}, {t}.", f"It's confirmed in Activity: {w}, {t}.",
                          f"I see it in Activity, confirmed: {w}, {t}.", f"Found it. {w} shows as confirmed, {t}.",
                          f"That one is confirmed. {w}, {t}."])
    elif r["status"] == "in_progress":
        out = rng.choice([f"It's still in progress: {w}, {t}.", f"Activity shows it as still in progress: {w}, {t}.",
                          f"Still in progress. {w}, {t}.", f"I see it, and it's still in progress: {w}, {t}.",
                          f"Not finished yet, it's in progress: {w}, {t}."])
    else:
        ask = rng.choice(["I can check your balances.", "I can check your balances if you want.",
                          "Want me to check your balances?"])
        out = rng.choice([f"It shows as failed. {w}, {t}. {ask}", f"It shows as failed: {w}, {t}. {ask}",
                          f"Activity shows it as failed. {w}, {t}. {ask}"])
    return out, [r]


def ans_balance(rng, res, sym=None):
    hold = res["holdings"]
    if sym:
        h = [x for x in hold if x["symbol"] == sym]
        if not h:
            return rng.choice([f"I don't see any {sym} in your wallet.", f"You don't hold any {sym} right now."])
        lines = "\n".join(f"- {x['amount']} {x['symbol']} on {CHAIN_NAME[x['chain']]}, ${fmt_usd(x['usd'])}" for x in h)
        return rng.choice([f"Your {sym}:", f"You hold {sym} here:", f"Here's your {sym}:"]) + "\n" + lines
    if not hold:
        return rng.choice(["Your wallet is empty.", "Your wallet has no holdings right now."])
    lines = "\n".join(f"- {x['amount']} {x['symbol']} on {CHAIN_NAME[x['chain']]}, ${fmt_usd(x['usd'])}" for x in hold)
    return rng.choice(["Here's what you hold:", "You hold:", "Your holdings:"]) + "\n" + lines


def fmt_usd(x):
    return f"{x:,.2f}"


HOWTO_ANS = [
    "Open Wallets and tap Activity. It lists your sends, swaps and buys with their status. Want me to show your recent activity here?",
    "Go to Wallets > Activity. Each item shows its status: confirmed, failed or in progress. I can also pull it up here if you want.",
    "Activity is in Wallets. Open it to see your sends, swaps and buys. Want me to show the latest here?",
    "In the app, open Wallets and then Activity. I can also read it out for you, just say so.",
    "It's in Wallets > Activity. You'll see each send, swap and buy with its status. Should I show your recent activity here?",
    "Tap Wallets, then Activity. I can show the latest items here too if you like.",
]
HASH_ANS = [
    "A transaction hash is a unique ID for one transaction on a blockchain. You can paste it into a block explorer to see its status and details.",
    "It's the unique ID of a transaction, like a receipt number. A block explorer lets you look it up and see where it stands.",
    "A tx hash is a long string that identifies a single transaction on-chain. Search it in a block explorer for its status, amount and addresses.",
    "It's a unique code for one transaction on a blockchain. Anyone can look it up in a block explorer.",
    "The hash is the transaction's ID. Paste it into a block explorer such as Etherscan to check where it stands.",
    "Think of it as a receipt number. It's a unique ID for one transaction, and a block explorer shows its details.",
]

# ------------------------------------------------------------------ scenarios
def new_world(rng, bio, rows_n=0, today=False, force_pending=False, wallet=True):
    w = sim.World(rng)
    device = "iPhone" if bio == "Face ID" else ("Mac" if rng.random() < 0.9 else "iPhone")
    metas = random_metas(rng, w, rows_n)
    times = gen_times(rng, rows_n, today)
    stat = assign_status(rng, times, force_pending)
    return w, device, metas, times, stat


def world_dict(device, activity, wallet=True, holdings=None):
    d = {"has_wallet": wallet, "device": device, "activity": activity}
    if holdings is not None:
        d["holdings"] = holdings
    return d


def holdings_v2(w):
    out = []
    for b in w.balances:
        out.append({"amount": fmt_dec(b["amount"]), "chain": "bnb" if b["chain"] == "bsc" else b["chain"],
                    "symbol": b["symbol"], "usd": b["usd"]})
    return out


def pick(rng, lst):
    return rng.choice(lst)


class Retry(Exception):
    pass


def finish(c, kind, world, extra=None, checks=None):
    whats = [r["what"] for r in world.get("activity", [])]
    if len(set(whats)) != len(whats):
        raise Retry("duplicate row text")
    meta = {"source": SRC, "generator": SRC, "kind": kind, "bio": c.bio, "lang": "en", "world": world,
            "call_worlds": c.call_worlds, "expect": c.expect, "checks": checks or [],
            "device": world.get("device")}
    if extra:
        meta.update(extra)
    return {"messages": c.msgs, "tools": True, "meta": meta}


def rows_n(rng, lo=1, hi=8, empty_p=0.0):
    if empty_p and rng.random() < empty_p:
        return 0
    return rng.randint(lo, hi)


def sc_general(rng, split, bio, T, multi=False):
    tpl = pick(rng, T["general"])
    today = "today" in tpl
    n = rows_n(rng, 1, 8, 0.09)
    if multi and n < 2:
        n = rng.randint(2, 5)
    fp = multi and True
    w, dev, metas, times, stat = new_world(rng, bio, n, today, force_pending=fp)
    if multi:
        # follow-ups need the whole list on screen
        for _ in range(30):
            if sum(s == "failed" for s in stat) or sum(s == "in_progress" for s in stat):
                break
            stat = assign_status(rng, times, True)
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {}))
    res = c.call("get_activity", {})
    return c, world, metas, res, "general"


def finish_general(c, rng, world, metas, res):
    rows = res["activity"]
    if not rows:
        c.say(ans_empty(rng, res))
        return [], "empty"
    text, shown = ans_list(rng, res)
    c.say(text)
    return shown, "list"


def build_general(rng, split, bio, T, multi):
    c, world, metas, res, _ = sc_general(rng, split, bio, T, multi)
    shown, sub = finish_general(c, rng, world, metas, res)
    first_user = c.msgs[1]["content"]
    extra = {"intent": "general", "sub": sub, "shown": shown, "limit": None, "rows_result": len(res["activity"]),
             "in_progress": res["in_progress"]}
    if not multi or sub == "empty":
        checks = ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"]
        return finish(c, "general_" + sub, world, extra, checks)
    # follow-up turn
    rows = res["activity"]
    p = res["in_progress"]
    f_all = [r for r in rows if r["status"] == "failed"]
    pend = [r for r in rows if r["status"] == "in_progress"]
    opts = []
    if pend:
        opts += ["pending", "refresh"]
    if f_all and len(rows) <= 5:
        opts += ["failed", "failed"]
    if not opts:
        return finish(c, "general_" + sub, world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])
    kind = rng.choice(opts)
    if kind == "pending":
        c.user(compose(rng, pick(rng, T["f_pending"]), {}))
        sel = pend
        c.say(rng.choice([f"{len(sel)} still in progress:", f"Still in progress:"]) + "\n" + "\n".join(line(r) for r in sel))
        extra.update({"intent": "followup_pending", "shown": sel})
        return finish(c, "followup_pending", world, extra, ["right_tool_new", "no_unbacked_done", "rows_verbatim"])
    if kind == "failed":
        c.user(compose(rng, pick(rng, T["f_failed"]), {}))
        sel = f_all
        lead = rng.choice(["It shows as failed:", "That one shows as failed:"]) if len(sel) == 1 else \
            rng.choice([f"{len(sel)} show as failed:", f"{len(sel)} of them show as failed:"])
        c.say(lead + "\n" + "\n".join(line(r) for r in sel) + "\n\n" +
              rng.choice(["I can check your balances.", "I can check your balances if you want.", "Want me to check your balances?"]))
        extra.update({"intent": "followup_failed", "shown": sel})
        return finish(c, "followup_failed", world, extra, ["right_tool_new", "no_unbacked_done", "rows_verbatim"])
    # refresh: one in-progress row resolves, same call again
    act2 = json.loads(json.dumps(world["activity"]))
    tgt = [i for i, r in enumerate(act2) if r["status"] == "in_progress"]
    flip = rng.choice(tgt)
    act2[flip]["status"] = "confirmed" if rng.random() < 0.7 else "failed"
    world2 = dict(world)
    world2["activity"] = act2
    c.world = world2
    c.user(compose(rng, pick(rng, T["f_refresh"]), {}))
    res2 = c.call("get_activity", {})
    text, shown2 = ans_list(rng, res2)
    c.say(text)
    extra.update({"intent": "followup_refresh", "shown": shown2, "in_progress": res2["in_progress"],
                  "rows_result": len(res2["activity"])})
    return finish(c, "followup_refresh", world2, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_limit(rng, split, bio, T):
    nlim = pick(rng, T["limits"])
    tpl = pick(rng, T["limit"])
    n_all = rng.randint(2, 8) if rng.random() > 0.05 else 0
    w, dev, metas, times, stat = new_world(rng, bio, n_all, False)
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {"n": nlim}))
    res = c.call("get_activity", {"limit": nlim})
    if not res["activity"]:
        c.say(ans_empty(rng, res))
        shown, sub = [], "empty"
    else:
        text, shown = ans_list(rng, res)
        c.say(text)
        sub = "list"
    extra = {"intent": "limit", "sub": sub, "shown": shown, "limit": nlim, "rows_result": len(res["activity"]),
             "in_progress": res["in_progress"]}
    return finish(c, "limit_" + sub, world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_pending(rng, split, bio, T):
    tpl = pick(rng, T["pending"])
    n = rows_n(rng, 1, 8, 0.07)
    w, dev, metas, times, stat = new_world(rng, bio, n, False, force_pending=(rng.random() < 0.75))
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {}))
    res = c.call("get_activity", {})
    if not res["activity"]:
        c.say(ans_empty(rng, res))
        shown, sub = [], "empty"
    else:
        text, shown = ans_pending(rng, res)
        c.say(text)
        sub = "pending" if res["in_progress"] else "none_pending"
    extra = {"intent": "pending", "sub": sub, "shown": shown, "limit": None, "rows_result": len(res["activity"]),
             "in_progress": res["in_progress"]}
    return finish(c, "pending_" + sub, world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_kind_pending(rng, split, bio, T):
    ukind = rng.choice(["swap", "swap", "bridge", "send", "buy", "deposit", "withdraw"])
    kp = pick(rng, KIND_NOUNS[ukind])
    kp1 = {"swap": "swap", "bridge": "bridge", "send": "send", "buy": "buy", "deposit": "deposit",
           "withdraw": "withdrawal"}[ukind]
    tpl = pick(rng, T["kind_pending"])
    want = rng.random() < 0.7
    for _ in range(100):
        n = rng.randint(1, 8)
        w, dev, metas, times, stat = new_world(rng, bio, n, False)
        if want:
            kind_pick = rng.choice(sorted(KIND_CLASS[ukind]))
            i = rng.randrange(min(n, 3))
            metas[i] = make_row(rng, w, kind_pick)
            stat[i] = "in_progress"
        hit = [j for j in match([m for _, m in metas], ukind) if stat[j] == "in_progress"]
        if bool(hit) == want:
            break
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {"kp": kp, "kp1": kp1}))
    res = c.call("get_activity", {})
    mm = [m for _, m in metas]
    text, shown = ans_kind_pending(rng, res, mm, ukind)
    c.say(text)
    extra = {"intent": "kind_pending", "sub": "match" if hit else "no_match", "ukind": ukind, "shown": shown,
             "limit": None, "rows_result": len(res["activity"]), "in_progress": res["in_progress"], "metas": mm}
    return finish(c, "kind_pending_" + extra["sub"], world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_latest(rng, split, bio, T):
    tpl = pick(rng, T["latest"])
    n = rows_n(rng, 1, 8, 0.07)
    w, dev, metas, times, stat = new_world(rng, bio, n, False)
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {}))
    res = c.call("get_activity", {})
    if not res["activity"]:
        c.say(ans_empty(rng, res))
        shown, sub = [], "empty"
    else:
        text, shown = ans_latest(rng, res)
        c.say(text)
        sub = "row"
    extra = {"intent": "latest", "sub": sub, "shown": shown, "limit": None, "rows_result": len(res["activity"]),
             "in_progress": res["in_progress"]}
    return finish(c, "latest_" + sub, world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_last_kind(rng, split, bio, T):
    ukind = rng.choice(["send", "send", "swap", "bridge", "buy", "deposit", "withdraw"])
    kw = pick(rng, KIND_SING[ukind])
    tpl = pick(rng, T["last_kind"])
    want = rng.random() < 0.85
    for _ in range(100):
        n = rng.randint(1, 8)
        w, dev, metas, times, stat = new_world(rng, bio, n, False)
        if want:
            i = rng.randrange(n)
            metas[i] = make_row(rng, w, rng.choice(sorted(KIND_CLASS[ukind])))
        mm = [m for _, m in metas]
        if bool(match(mm, ukind)) == want:
            break
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {"kw": kw}))
    res = c.call("get_activity", {})
    text, shown = ans_last_kind(rng, res, mm, ukind)
    c.say(text)
    extra = {"intent": "last_kind", "sub": "match" if want else "no_match", "ukind": ukind, "shown": shown,
             "limit": None, "rows_result": len(res["activity"]), "in_progress": res["in_progress"], "metas": mm}
    return finish(c, "last_kind_" + extra["sub"], world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_specific(rng, split, bio, T):
    ukind = rng.choice(["send", "swap", "bridge", "buy", "buy", "deposit", "withdraw", "send"])
    form = rng.choice(["symamt", "sym", "sym", "none", "symamt"])
    if ukind in ("bridge", "swap") and form == "none" and rng.random() < 0.3:
        form = "sym"
    syms = T["syms"][ukind]
    sym = rng.choice(syms)
    want = rng.random() < 0.85
    near_miss = (not want) and form == "symamt" and rng.random() < 0.5
    sub = None
    for _ in range(300):
        n = rng.randint(1, 8)
        w, dev, metas, times, stat = new_world(rng, bio, n, False)
        target_kind = ukind
        if ukind == "swap" and rng.random() < 0.3:
            target_kind = "bridge"
        what_t, meta_t = make_row(rng, w, target_kind, sym)
        amt = None
        amt_sym = sym
        if form.startswith("symamt"):
            if ukind == "buy":
                use_usd = rng.random() < 0.55
                amt = meta_t["amts"][1] if use_usd else meta_t["amts"][0]
                formkey = "symamt_usd" if use_usd else "symamt_tok"
            else:
                amt = meta_t["amts"][0]
                formkey = "symamt"
        else:
            formkey = form
        sy = None if form == "none" else sym
        idx_t = None
        if want:
            idx_t = rng.randrange(n)
            metas[idx_t] = (what_t, meta_t)
        elif near_miss:
            other = make_row(rng, w, target_kind, sym)
            if other[1]["amts"][0] == meta_t["amts"][0]:
                continue
            metas[rng.randrange(n)] = other
        mm = [m for _, m in metas]
        hits = match(mm, ukind, sy, amt)
        if (want and len(hits) == 1) or (not want and not hits):
            break
    else:
        raise RuntimeError("specific world not found")
    if want:
        tstat = rng.choices(["confirmed", "in_progress", "failed"], [4, 3, 3])[0]
        if tstat == "in_progress" and idx_t >= min(n, 3):
            j = rng.randrange(min(n, 3))
            metas[idx_t], metas[j] = metas[j], metas[idx_t]
            idx_t = j
        stat[idx_t] = tstat
    act = build_activity(metas, times, stat)
    world = world_dict(dev, act)
    c = Conv(rng, world, bio)
    thing_tpl = pick(rng, T["things"][(ukind, formkey)])
    tpl = pick(rng, T["status"]).replace("{thing}", thing_tpl)
    c.user(compose(rng, tpl, {"sym": sym, "amt": amt}))
    res = c.call("get_activity", {})
    if not res["activity"]:
        c.say(ans_empty(rng, res))
        shown, sub = [], "empty"
    else:
        text, shown = ans_specific(rng, res, mm, ukind, sy, amt)
        c.say(text)
        sub = ("hit_" + shown[0]["status"]) if shown else "no_match"
    extra = {"intent": "specific", "sub": sub, "ukind": ukind, "sym": sy, "amt": amt, "shown": shown, "limit": None,
             "rows_result": len(res["activity"]), "in_progress": res["in_progress"], "metas": mm}
    return finish(c, "specific_" + sub, world, extra, ["right_tool_new", "exact_digits", "no_unbacked_done", "rows_verbatim"])


def build_error(rng, split, bio, T):
    tpl = pick(rng, T["general"] + T["pending"])
    device = "iPhone" if bio == "Face ID" else ("Mac" if rng.random() < 0.9 else "iPhone")
    world = world_dict(device, [], wallet=False)
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {}))
    res = c.call("get_activity", {})
    c.say(ans_error(rng, res))
    extra = {"intent": "error", "sub": "error", "shown": [], "limit": None, "rows_result": 0, "in_progress": 0}
    return finish(c, "error_wallet", world, extra, ["right_tool_new", "no_unbacked_done"])


def build_balance(rng, split, bio, T):
    w = sim.World(rng)
    sym = None
    if rng.random() < 0.4:
        sym = rng.choice(T["bal_syms"])
        have = {b["symbol"] for b in w.balances}
        if sym not in have and rng.random() < 0.6 and w.balances:
            ch = rng.choice(["ethereum", "base", "arbitrum", "solana"])
            amt = round(rng.uniform(0.5, 40) if sym in sim.STABLES or w.prices[sym] < 100 else rng.uniform(0.01, 2), 4)
            w.balances.append({"symbol": sym, "chain": ch, "amount": amt, "usd": round(amt * w.prices[sym], 2)})
        tpl = pick(rng, T["balance_sym"])
    else:
        tpl = pick(rng, T["balance"])
    device = "iPhone" if bio == "Face ID" else "Mac"
    world = world_dict(device, [], holdings=holdings_v2(w))
    c = Conv(rng, world, bio)
    c.user(compose(rng, tpl, {"sym": sym}))
    res = c.call("get_balances", {})
    c.say(ans_balance(rng, res, sym))
    extra = {"intent": "balance", "sub": "balance", "shown": [], "limit": None, "rows_result": 0, "in_progress": 0, "sym": sym}
    return finish(c, "not_activity_balance", world, extra, ["right_tool_new", "no_unbacked_done"])


def build_plain(rng, split, bio, T, which):
    device = "iPhone" if bio == "Face ID" else "Mac"
    world = world_dict(device, [])
    c = Conv(rng, world, bio)
    if which == "howto":
        c.user(compose(rng, pick(rng, T["howto"]), {}))
        c.say(rng.choice(HOWTO_ANS))
        kind = "not_activity_howto"
    else:
        c.user(compose(rng, pick(rng, T["hash"]), {}))
        c.say(rng.choice(HASH_ANS))
        kind = "not_activity_hash"
    extra = {"intent": which, "sub": "plain", "shown": [], "limit": None, "rows_result": 0, "in_progress": 0}
    return finish(c, kind, world, extra, ["restraint", "no_unbacked_done"])


F_PENDING = ["which one is still pending", "which one's in progress", "what's the pending one", "the one in progress, what is it",
             "which is still going", "which one hasn't finished", "what's still in progress", "which one is the pending one"]
F_FAILED = ["which one failed", "what failed", "the failed one?", "which one is the failed one", "which ones failed",
            "what's the failed one", "which one didn't work", "show me the one that failed"]
F_REFRESH = ["check again", "any change?", "refresh", "check again please", "has anything changed", "look again",
             "anything new now?", "update?", "try again", "ok check it once more"]

INTENTS = [  # (name, weight)
    ("general", 40), ("multi", 12), ("limit", 36), ("pending", 26), ("kind_pending", 20), ("latest", 11),
    ("last_kind", 18), ("specific", 52), ("balance", 9), ("howto", 7), ("hash", 6), ("error", 6),
]
BUILD = {"general": lambda r, s, b, T: build_general(r, s, b, T, False),
         "multi": lambda r, s, b, T: build_general(r, s, b, T, True),
         "limit": build_limit, "pending": build_pending, "kind_pending": build_kind_pending, "latest": build_latest,
         "last_kind": build_last_kind, "specific": build_specific, "balance": build_balance,
         "howto": lambda r, s, b, T: build_plain(r, s, b, T, "howto"),
         "hash": lambda r, s, b, T: build_plain(r, s, b, T, "hash"), "error": build_error}


def allocate(n):
    tot = sum(w for _, w in INTENTS)
    raw = {k: n * w / tot for k, w in INTENTS}
    cnt = {k: max(1, int(v)) for k, v in raw.items()}
    while sum(cnt.values()) < n:
        k = max(raw, key=lambda k: raw[k] - cnt[k])
        cnt[k] += 1
    while sum(cnt.values()) > n:
        k = max(cnt, key=lambda k: cnt[k] - raw[k])
        cnt[k] -= 1
    return cnt


def pools(split):
    T = {
        "general": split_pool(GENERAL, split), "limit": split_pool(LIMIT, split), "pending": split_pool(PENDING, split),
        "kind_pending": split_pool(KIND_PENDING, split), "latest": split_pool(LATEST, split),
        "last_kind": split_pool(LAST_KIND, split), "status": split_pool(STATUS, split),
        "balance": split_pool(BALANCE, split), "balance_sym": split_pool(BALANCE_SYM, split),
        "howto": split_pool(HOWTO, split), "hash": split_pool(HASH, split),
        "f_pending": split_pool(F_PENDING, split), "f_failed": split_pool(F_FAILED, split),
        "f_refresh": split_pool(F_REFRESH, split),
        "things": {k: split_pool(v, split) for k, v in THINGS.items()},
        "syms": TRAIN_SYMS if split == "train" else EVAL_SYMS,
        "limits": TRAIN_LIMITS if split == "train" else EVAL_LIMITS,
        "bal_syms": TRAIN_BAL_SYMS if split == "train" else EVAL_BAL_SYMS,
    }
    for k, v in T.items():
        if isinstance(v, list):
            assert v, f"empty pool {k}"
    for k, v in T["things"].items():
        assert v, f"empty thing pool {k}"
    return T


def leak_index():
    """normalised user texts of every eval2 item (read only)."""
    idx = set()
    d = os.path.join(TRAIN_DIR, "eval2")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jsonl"):
            continue
        for ln in open(os.path.join(d, f)):
            o = json.loads(ln)
            for m in o.get("context", []) or []:
                if m.get("role") == "user":
                    idx.add(norm(m["content"]))
            for t in o.get("turns", []) or []:
                idx.add(norm(t))
            if isinstance(o.get("prompt"), str):
                idx.add(norm(o["prompt"]))
            if isinstance(o.get("question"), str):
                idx.add(norm(o["question"]))
    return idx


def last_user_pair(msgs):
    return [m["content"] for m in msgs if m["role"] == "user"]


def generate(seed, n, split, banned_norms):
    rng = random.Random(seed)
    T = pools(split)
    plan = []
    for k, c in allocate(n).items():
        plan += [k] * c
    rng.shuffle(plan)
    out, seen = [], set()
    drops = Counter()
    for name in plan:
        for attempt in range(80):
            child = random.Random(rng.getrandbits(64))
            bio = "Face ID" if child.random() < 0.7 else "Touch ID"
            try:
                ex = BUILD[name](child, split, bio, T)
            except Retry:
                drops[name + ":dup_rows"] += 1
                continue
            users = [norm(u) for u in last_user_pair(ex["messages"])]
            if any(u in seen or u in banned_norms for u in users):
                drops[name] += 1
                continue
            seen.update(users)
            ex["meta"]["template_split"] = split
            ex["meta"]["seed"] = seed
            out.append(ex)
            break
        else:
            raise RuntimeError(f"could not generate a unique {name}")
    return out, drops


# ------------------------------------------------------------------ eval items
def to_eval(ex, idx):
    m = ex["meta"]
    msgs = ex["messages"]
    last_u = max(i for i, x in enumerate(msgs) if x["role"] == "user")
    ctx, ref = msgs[:last_u + 1], msgs[last_u + 1:]
    calls = [e["calls"][0] for e in m["expect"] if "calls" in e]
    shown_what = [r["what"] for r in m["shown"]]
    ac = {
        "calls": calls,
        "no_call": not calls,
        "limit": m["limit"],
        "must_contain": shown_what,
        "in_progress": m["in_progress"],
        "max_list_lines": min(5, m["rows_result"]),
        "must_not_regex": BANNED_RE_STRINGS,
        "rows_in_result": m["rows_result"],
    }
    if m["intent"] in ("specific",):
        ac["expect_status"] = {"hit_confirmed": "confirmed", "hit_in_progress": "in progress", "hit_failed": "failed"}.get(m["sub"])
        ac["no_match_text"] = "I don't see it in Activity." if m["sub"] == "no_match" else None
    if m["sub"] == "empty":
        ac["empty_text"] = "No activity yet"
    if m["sub"] == "error":
        ac["error_sentence"] = True
    if any(r["status"] == "failed" for r in m["shown"]):
        ac["must_say"] = ["failed", "check your balances"]
    item = {"id": f"activity_tool-{idx}", "kind": m["kind"], "tools": True, "bio": m["bio"], "context": ctx,
            "reference": ref, "expect": m["expect"], "world": m["world"], "checks": m["checks"], "answer_checks": ac,
            "meta": {"generator": SRC, "intent": m["intent"], "call_worlds": m["call_worlds"], "device": m.get("device")}}
    return item


# ------------------------------------------------------------------ self-check
BANNED_RE_STRINGS = [
    r"\bRelay\b", r"went through", r"gone through", r"has been (sent|swapped|bought|deposited|withdrawn|closed)",
    r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed)\b", r"successfully",
    r"funds (are|have) (in|arrived)", r"^\s*done\b", r"not financial advice", "[·•]",
    r"what('s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you)",
    r"can't see the future", r"\ball set\b", r"should have", r"\bprobably\b", r"\blikely\b", r"\bmaybe\b",
    r"\bbecause (it|the|of)\b", r"I think", r"\bcompleted\b", r"\bdone\b",
]
BANNED_RE = [re.compile(p, re.I) for p in BANNED_RE_STRINGS]


def strip_rows(text, rows):
    for r in sorted(rows, key=lambda r: -len(r["what"])):
        text = text.replace(r["what"], " ")
        text = text.replace(r["time"], " ")
    return text


def render_check(msgs, bio):
    """Render with the 19-tool block and parse every tool call back; return an estimate of tokens."""
    block = H.TOOLS_BLOCK
    saved = H.TOOLS_BLOCK
    H.TOOLS_BLOCK = block.replace("Face ID", bio)
    try:
        text = H.render_text(msgs, True)
    finally:
        H.TOOLS_BLOCK = saved
    for m in msgs:
        if m["role"] == "assistant":
            t = H.assistant_text(m)
            calls, bad = H.parse_calls(t)
            want = [{"name": c["name"], "arguments": c["arguments"]} for c in (m.get("tool_calls") or [])]
            assert bad == 0 and calls == want, "render round trip failed"
    return len(text) / 3.3


def schema_check(call):
    name, args = call["name"], call["arguments"]
    assert name in TOOL_SCHEMA, f"unknown tool {name}"
    sch = TOOL_SCHEMA[name]
    for k in sch.get("required", []):
        assert k in args, f"{name} missing {k}"
    for k, v in args.items():
        assert k in sch["properties"], f"{name} unknown arg {k}"
        ty = sch["properties"][k]["type"]
        if ty == "integer":
            assert isinstance(v, int) and not isinstance(v, bool), f"{name}.{k} not an integer"
        elif ty == "number":
            assert isinstance(v, (int, float)) and not isinstance(v, bool)
        elif ty == "string":
            assert isinstance(v, str)


def check_record(ex, who):
    m = ex["meta"]
    msgs = ex["messages"]
    assert ex["tools"] is True and m["source"] == SRC
    bio, other = m["bio"], ("Touch ID" if m["bio"] == "Face ID" else "Face ID")
    blob = json.dumps(msgs, ensure_ascii=False)
    assert "{BIO}" not in blob and other not in blob, f"{who}: bio word mismatch"
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(SYSTEM_V2.split("{BIO}")[0]), f"{who}: system prompt"
    assert bio in msgs[0]["content"]
    # structure: user, (assistant calls, tool)*, assistant text
    ci = 0
    i = 1
    last_world = None
    while i < len(msgs):
        assert msgs[i]["role"] == "user", f"{who}: expected user at {i}"
        utext = msgs[i]["content"]
        i += 1
        calls_here = []
        while i < len(msgs) and msgs[i]["role"] == "assistant" and msgs[i].get("tool_calls"):
            tc = msgs[i]["tool_calls"]
            assert len(tc) == 1 and msgs[i]["content"] == ""
            schema_check(tc[0])
            assert msgs[i + 1]["role"] == "tool"
            res_s = msgs[i + 1]["content"]
            res = json.loads(res_s)
            replay = call_tool(m["call_worlds"][ci], tc[0]["name"], tc[0]["arguments"])
            assert json.dumps(replay, ensure_ascii=False) == res_s, f"{who}: replay mismatch"
            calls_here.append((tc[0], res))
            ci += 1
            i += 2
        assert i < len(msgs) and msgs[i]["role"] == "assistant" and not msgs[i].get("tool_calls"), f"{who}: missing answer"
        ans = msgs[i]["content"]
        i += 1
        digits = re.findall(r"\d+", utext)
        for call, res in calls_here:
            if call["name"] == "get_activity":
                lim = call["arguments"].get("limit")
                intent = m["intent"]
                if "limit" in call["arguments"]:
                    assert intent == "limit" and digits == [str(lim)], f"{who}: limit without matching digit"
                    assert isinstance(lim, int) and lim >= 1
                else:
                    assert intent != "limit", f"{who}: limit intent without limit"
                    if intent in ("general", "pending", "kind_pending", "latest", "last_kind", "error", "followup_refresh"):
                        assert not digits, f"{who}: digit in a no-limit user text"
            else:
                assert call["name"] == "get_balances" and call["arguments"] == {}
        # answer checks
        if calls_here:
            call, res = calls_here[-1]
        else:
            call, res = None, None
        assert 3 <= len(ans) < 900
        for rx in BANNED_RE:
            rows_all = res.get("activity", []) if res else []
            if rx.search(strip_rows(ans, rows_all)):
                raise AssertionError(f"{who}: banned phrase {rx.pattern!r} in {ans!r}")
        assert len(ans.split()) <= 110, f"{who}: answer too long"
        if call and call["name"] == "get_activity" and "activity" in res:
            rows = res["activity"]
            # in_progress consistent
            assert res["in_progress"] == sum(1 for r in rows if r["status"] == "in_progress")
            assert [r["time"] for r in rows] == sorted([r["time"] for r in rows], reverse=True), f"{who}: not newest first"
            for r in rows:
                assert set(r) == {"status", "time", "what"} and r["status"] in STATUS_TXT
            # every line copied from a row, status/time matching
            lines = [l for l in ans.split("\n") if l.startswith("- ")]
            assert len(lines) <= min(5, len(rows)) or m["intent"] in ("followup_failed",), f"{who}: list longer than result"
            known = {line(r): r for r in rows}
            for l in lines:
                assert l in known, f"{who}: line not from a row: {l!r}"
            # row text in the answer must be verbatim in the result; status words consistent per segment
            for seg in re.split(r"\n", ans):
                present = [r for r in rows if r["what"] in seg]
                if not present:
                    continue
                words = set(re.findall(r"confirmed|failed|in progress", strip_rows(seg, rows)))
                allowed = {STATUS_TXT[r["status"]] for r in present}
                assert words == allowed, f"{who}: status mismatch in {seg!r}"
            # numbers: only counts computed from the result, numbers of the user text, or inside row text
            rest = strip_rows(ans, rows)
            nums = re.findall(r"\d+", rest)
            allowed_nums = {str(len(rows)), str(res["in_progress"]), str(min(5, len(rows))), "1"}
            allowed_nums |= {str(sum(1 for r in rows if r["status"] == "failed")), str(sum(1 for r in rows if r["status"] == "in_progress"))}
            allowed_nums |= {str(sum(1 for r in rows[:5] if r["status"] == "failed"))}
            allowed_nums |= set(digits)
            if m["intent"] in ("kind_pending", "pending", "followup_pending", "followup_failed"):
                allowed_nums |= {str(k) for k in range(0, 10)} & {str(x) for x in range(0, len(rows) + 1)}
            for x in nums:
                assert x in allowed_nums, f"{who}: unexplained number {x} in {ans!r}"
            # headline counts
            mm = re.search(r"(\d+) recent items?", ans)
            if mm:
                assert int(mm.group(1)) == len(rows), f"{who}: wrong item count"
            for mm in re.finditer(r"(One|\d+)( of them)? shows? as failed|(\d+) show as failed", ans):
                pass
            f_shown = sum(1 for r in rows[:5] if r["status"] == "failed")
            if "\n\n" in ans and "failed" in ans.split("\n\n")[-1].split("\n")[0] and m["intent"] in ("general", "limit", "followup_refresh"):
                tail = ans.split("\n\n")[-1]
                mf = re.match(r"(It|One|One of them|\d+|\d+ of them) shows? as failed|(\d+)( of them)? show as failed", tail)
                if mf:
                    first = tail.split(" ")[0]
                    if first.isdigit():
                        assert int(first) == f_shown, f"{who}: failed count"
                    else:
                        assert f_shown == 1, f"{who}: failed count"
            # failed rows shown must be flagged, and "shows as failed" only if a failed row is shown
            shown_failed = any(r["status"] == "failed" for r in rows if r["what"] in ans)
            if "shows as failed" in ans or "show as failed" in ans:
                assert shown_failed, f"{who}: failed claim without a failed row"
            if shown_failed:
                assert "failed" in strip_rows(ans, rows) and "check your balances" in ans, f"{who}: failed row not flagged"
            if not rows:
                assert ans.startswith("No activity yet") or ans.startswith("There's no activity yet")
            # specific / no-match gold
            if m["intent"] == "specific":
                hits = match(m["metas"][:len(rows)], m["ukind"], m["sym"], m["amt"])
                if hits:
                    assert rows[hits[0]]["what"] in ans, f"{who}: specific row missing"
                    assert len(hits) == 1
                elif rows:
                    assert ans.startswith("I don't see it in Activity."), f"{who}: no-match text"
            if m["intent"] == "last_kind":
                hits = match(m["metas"][:len(rows)], m["ukind"])
                if hits:
                    assert rows[hits[0]]["what"] in ans
                elif rows:
                    assert ans.startswith("I don't see it in Activity.")
            if m["intent"] == "latest" and rows:
                assert rows[0]["what"] in ans
            if m["intent"] == "pending" and rows:
                pend = [r for r in rows if r["status"] == "in_progress"]
                for r in pend:
                    assert r["what"] in ans
                if not pend:
                    assert "Nothing" in ans or "nothing" in ans or "No, nothing" in ans
        if call and call["name"] == "get_balances":
            for h in res["holdings"]:
                pass
            nums = re.findall(r"\d[\d,]*\.?\d*", ans)
            src = json.dumps(res) + utext
            for x in nums:
                x2 = x.rstrip(".")
                ok = x2 in src or x2.replace(",", "") in src or any(abs(float(x2.replace(",", "")) - h["usd"]) < 0.006 for h in res["holdings"])
                assert ok, f"{who}: balance number {x} not in result"
        if m["intent"] in ("howto", "hash"):
            assert not calls_here, f"{who}: plain item called a tool"
            assert "Wallets" in ans or m["intent"] == "hash"
    # expect list equals what happened in the final turn
    n_calls_final = len([e for e in m["expect"] if "calls" in e])
    assert m["expect"][-1] == {"text": True}
    # render
    est = render_check(msgs, bio)
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--eval-seed", type=int, default=9313)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(TRAIN_DIR, "data3", "activity_tool.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(TRAIN_DIR, "eval3", "activity_tool.jsonl"))
    a = ap.parse_args()
    assert a.seed != a.eval_seed
    leaks = leak_index()
    # eval first so train can be filtered against it, then train; both avoid eval2 texts
    evl_ex, ed = generate(a.eval_seed, a.eval_n, "eval", leaks)
    eval_norms = {norm(u) for ex in evl_ex for u in last_user_pair(ex["messages"])}
    train, td = generate(a.seed, a.n, "train", leaks | eval_norms)
    train_norms = {norm(u) for ex in train for u in last_user_pair(ex["messages"])}
    assert not (train_norms & eval_norms)
    items = [to_eval(ex, i) for i, ex in enumerate(evl_ex)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # self-check
    failures = []
    max_tok = 0
    try:
        rows_t = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        rows_e = [json.loads(l) for l in open(a.eval_out, encoding="utf-8")]
        assert len(rows_t) == len(train) and len(rows_e) == len(items)
        seen = {}
        for i, ex in enumerate(rows_t):
            max_tok = max(max_tok, check_record(ex, f"train[{i}]"))
            for u in last_user_pair(ex["messages"]):
                assert norm(u) not in seen, f"duplicate train user text: {u!r}"
                seen[norm(u)] = i
                assert norm(u) not in leaks, f"train text in eval2: {u!r}"
        seen_e = set()
        for i, (it, ex) in enumerate(zip(rows_e, evl_ex)):
            for k in ("id", "kind", "tools", "bio", "context", "reference", "expect", "world", "checks", "answer_checks"):
                assert k in it, f"eval[{i}] missing {k}"
            max_tok = max(max_tok, check_record(ex, it["id"]))
            full = it["context"] + it["reference"]
            assert full == ex["messages"], f"eval[{i}] context+reference != conversation"
            assert it["context"][-1]["role"] == "user"
            for c in it["answer_checks"]["calls"]:
                schema_check(c)
            for u in last_user_pair(it["context"]):
                n_ = norm(u)
                assert n_ not in seen and n_ not in leaks and n_ not in seen_e, f"eval text not disjoint: {u!r}"
                seen_e.add(n_)
            ans = it["reference"][-1]["content"]
            for w in it["answer_checks"]["must_contain"]:
                assert w in ans, f"eval[{i}] gold misses {w!r}"
        # held-out template/slot check
        for ex in train:
            assert ex["meta"]["template_split"] == "train"
        for ex in evl_ex:
            assert ex["meta"]["template_split"] == "eval"
    except AssertionError as e:
        failures.append(str(e))
    kinds_t = Counter(ex["meta"]["intent"] for ex in train)
    kinds_e = Counter(ex["meta"]["intent"] for ex in evl_ex)
    bios = Counter(ex["meta"]["bio"] for ex in train)
    print("train intents:", dict(kinds_t), file=sys.stderr)
    print("eval intents:", dict(kinds_e), file=sys.stderr)
    print("bio:", dict(bios), " est max tokens:", int(max_tok), " retries:", dict(td), file=sys.stderr)
    if failures:
        print(f"activity_tool: {len(train)} train, {len(items)} eval, self-check FAILED: {failures[0]}")
        sys.exit(1)
    print(f"activity_tool: {len(train)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
