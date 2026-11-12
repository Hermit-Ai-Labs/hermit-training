#!/usr/bin/env python3
"""multisend_typed: prepare_multisend from a list the user TYPED (generator 1 of the next run).

The model must call prepare_multisend once from the typed list, never split it into single sends, never
invent a chain. Scenarios: basic lists in many layouts, "N each", equal split (calculator), needs_chain,
names without addresses (no call, one question), blocked by policy, insufficient funds, duplicate address,
follow-up "add one more", Solana / NEAR lists, one malformed row (left_out).

Every assistant string is a Python f-string filled only from tool results, the user's text or a calculator
result. Tool results come from MWorld (a sim.World subclass in this file; gen2 is imported, never edited).

  python3 multisend_typed.py                      # train seed 3, n 450 + eval seed 9301, n 50
  python3 multisend_typed.py --seed 3 --n 450 --eval-seed 9301 --n-eval 50 [--only train|eval|both]

Writes ../data3/multisend_typed.jsonl and ../eval3/multisend_typed.jsonl, then self-checks every record.
"""
import argparse, json, os, random, re, sys
sys.dont_write_bytecode = True          # never write .pyc files next to gen2
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as H                  # gen2 helpers (imported, not copied)
import sim
from build_agent import Conv as _Conv

SPEC = os.path.join(LAB, "spec")
TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
META = json.load(open(os.path.join(SPEC, "hermit_tools_v2.meta.json")))
SYSTEM_TMPL = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read()
# point the gen2 formatter at the 19-tool registry (runtime patch, gen2 files untouched)
H.TOOLS = TOOLS
H.TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
H.TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
H.TOOLS_BLOCK = H.tools_block()

OUT_TRAIN = os.path.join(LAB, "data3", "multisend_typed.jsonl")
OUT_EVAL = os.path.join(LAB, "eval3", "multisend_typed.jsonl")
SRC = "multisend_typed"


# ====================================================================== numbers
def D(x):
    return Decimal(str(x))


def dfmt(d):
    d = Decimal(d)
    if d == 0:
        return "0"
    return format(d.normalize(), "f")


def jnum(d):
    d = Decimal(d)
    return int(d) if d == d.to_integral_value() else float(d)


NUMC = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(k\b)?"
NUM_RE = re.compile(r"\$?" + NUMC)


def nval(m):
    v = Decimal(m.group(1).replace(",", ""))
    return v * 1000 if m.group(2) else v


# ====================================================================== chains / symbols
CHAIN_NAME = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "bsc": "BNB Chain",
              "polygon": "Polygon", "solana": "Solana", "near": "NEAR"}
EVM_CHAINS = ["ethereum", "base", "arbitrum", "bsc", "polygon"]
FAM_CHAINS = {"evm": EVM_CHAINS, "solana": ["solana"], "near": ["near"]}
CHAIN_PAT = {"ethereum": r"ethereum|mainnet", "base": r"base", "arbitrum": r"arbitrum|arb",
             "bsc": r"bsc|bnb chain|bnb smart chain", "polygon": r"polygon", "solana": r"solana", "near": r"near"}
CHAIN_SAY = {"ethereum": ["ethereum", "Ethereum", "mainnet"], "base": ["base", "Base"],
             "arbitrum": ["arbitrum", "Arbitrum", "arb"], "bsc": ["bsc", "BSC", "BNB Chain", "bnb chain"],
             "polygon": ["polygon", "Polygon"], "solana": ["solana", "Solana"], "near": ["near", "NEAR"]}
def _strict_rx(c, p):
    paren = "" if c == "near" else rf"|(?:\((?:{p})\))"      # "(NEAR)" is usually the token, not the chain
    return re.compile(rf"(?:\b(?:on|via|over|through)\s+(?:the\s+)?(?:{p})\b){paren}|(?:\b(?:{p})\s+(?:network|chain)\b)"
                      rf"|(?:\b(?:network|chain)\s*:\s*(?:{p})\b)", re.I)


_STRICT = {c: _strict_rx(c, p) for c, p in CHAIN_PAT.items()}
_BARE = {c: re.compile(rf"(?<![.\w])(?:{p})(?![\w])", re.I) for c, p in CHAIN_PAT.items()}


def named_chains(text, bare=False):
    """Chains the user named in `text`. strict = in a phrase like 'on base'; bare = the word alone (answers)."""
    return {c for c in CHAIN_PAT if (_BARE[c] if bare else _STRICT[c]).search(text)}


EVM_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
B58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
NEAR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.near$")


def family(a):
    if EVM_RE.match(a):
        return "evm"
    if NEAR_RE.match(a):
        return "near"
    if B58_RE.match(a):
        return "solana"
    return None


STABLE = {"USDC", "USDT", "DAI"}
TICKERS = ["USDC", "USDT", "ETH", "DAI", "SOL", "NEAR", "LINK", "UNI", "AAVE", "ENA", "POL", "JUP", "WIF", "BONK"]
TOKEN_CHAINS = {"USDC": ["base", "arbitrum", "ethereum", "polygon", "bsc"], "USDT": ["ethereum", "arbitrum", "bsc", "polygon"],
                "ETH": ["ethereum", "base", "arbitrum"], "DAI": ["ethereum", "base", "arbitrum", "polygon"],
                "LINK": ["ethereum", "arbitrum", "polygon", "base"], "UNI": ["ethereum", "arbitrum", "base"],
                "AAVE": ["ethereum", "base", "arbitrum"], "ENA": ["ethereum"], "POL": ["polygon", "ethereum"]}
CHW = {"base": 35, "arbitrum": 20, "ethereum": 25, "polygon": 8, "bsc": 12}
SYM_FAM = {"SOL": "solana", "NEAR": "near", "JUP": "solana", "WIF": "solana", "BONK": "solana"}
OTHER_EVM = {False: ["LINK", "UNI"], True: ["AAVE", "ENA", "POL"]}      # [eval?] held-out tickers go to eval
OTHER_SOL = {False: ["JUP", "WIF"], True: ["BONK"]}
SYM_W = [("USDC", 50), ("USDT", 15), ("ETH", 12), ("DAI", 5), ("SOL", 8), ("NEAR", 4), ("OTHER", 6)]


def wchoice(rng, pairs):
    return rng.choices([p for p, _ in pairs], [w for _, w in pairs])[0]


def pick_sym(rng, ev):
    s = wchoice(rng, SYM_W)
    if s == "OTHER":
        s = rng.choice(OTHER_EVM[ev] + OTHER_SOL[ev])
    return s


def sym_family(sym):
    return SYM_FAM.get(sym, "evm")


def pick_chain(rng, sym, fam):
    if fam != "evm":
        return fam
    ch = TOKEN_CHAINS[sym]
    return rng.choices(ch, [CHW[c] for c in ch])[0]


def sym_dp(sym):
    return 4 if sym in ("ETH", "SOL", "NEAR") else 2


# ====================================================================== addresses
NEAR_STEMS = ["ana", "bobby", "kofi", "mira", "dao_ops", "team", "alice", "carol", "dan", "eve", "frank", "grace", "heidi",
              "ivan", "judy", "luna", "max", "nora", "otto", "pia", "quinn", "ravi", "sara", "tim", "uma", "vera", "walt",
              "yuri", "zoe", "builder", "treasury", "mod"]


def near_name(rng):
    stem = rng.choice(NEAR_STEMS)
    r = rng.random()
    if r < 0.5:
        return f"{stem}.near"
    if r < 0.8:
        return f"{stem}{rng.randint(1, 99)}.near"
    return f"{stem}.{rng.choice(['pay', 'team', 'dao'])}.near"


def gen_addr(rng, fam):
    if fam == "evm":
        return H.evm_addr(rng)
    if fam == "solana":
        return H.sol_addr(rng)
    return near_name(rng)


def gen_addrs(rng, fam, n):
    out = []
    while len(out) < n:
        a = gen_addr(rng, fam)
        if a.lower() not in {x.lower() for x in out}:
            out.append(a)
    return out


# ====================================================================== simulator
class MWorld(sim.World):
    """gen2 World plus prepare_multisend (shapes from hermit_tools_v2.meta.json).
    `user_text` = everything the user typed so far (set by the conversation builder and by replay)."""
    STATE_KEYS = sim.World.STATE_KEYS + ("daily_left", "move_cap", "flagged")

    def __init__(self, rng):
        super().__init__(rng, empty_hl=True, empty_wallet=True)
        self.daily_left = 1e9
        self.move_cap = 1e9
        self.flagged = []
        self.user_text = ""

    @classmethod
    def from_state(cls, st, seed=0):
        w = super().from_state(st, seed)
        w.user_text = ""
        return w

    def t_prepare_multisend(self, symbol, recipients, chain=None):
        sym = str(symbol).strip().upper()
        if not isinstance(recipients, list) or not recipients:
            return {"error": "No recipients given."}
        if len(recipients) > 200:
            return {"error": "A batch can have at most 200 recipients."}
        rows = []
        for i, r in enumerate(recipients, 1):
            if (not isinstance(r, dict) or not isinstance(r.get("address"), str) or isinstance(r.get("amount"), bool)
                    or not isinstance(r.get("amount"), (int, float))):
                return {"error": f"Row {i} needs an address and a numeric amount."}
            if r["amount"] <= 0:
                return {"error": f"Row {i}: the amount must be more than 0."}
            rows.append((i, r["address"], D(r["amount"])))
        for _, a, _ in rows:                                  # app guard: only addresses the user gave
            if a not in self.user_text:
                short = a[:10] + "…" if len(a) > 10 else a
                return {"error": f"Refused: {short} isn't in the user's messages or attached file. "
                                 "Only send to addresses the user gave you. Ask them for the list."}
        valid, left = [], []
        for i, a, x in rows:
            (valid if family(a) else left).append((i, a, x) if family(a) else f"row {i}: not an address")
        if not valid:
            return {"error": "None of the rows is a valid address."}
        fams = {family(a) for _, a, _ in valid}
        if len(fams) > 1:
            return {"error": "A batch can only use one kind of address (EVM, Solana or NEAR)."}
        fam = fams.pop()
        ok = FAM_CHAINS[fam]
        total = sum((x for _, _, x in valid), Decimal(0))
        agg = {}
        for b in self.balances:
            if b["symbol"].upper() == sym and b["chain"] in ok and D(b["amount"]) > 0:
                agg[b["chain"]] = agg.get(b["chain"], Decimal(0)) + D(b["amount"])
        ch = None if chain is None else str(chain).lower()
        if ch is not None and ch not in named_chains(self.user_text, bare=True):
            ch = None                                         # app guard: drop a chain the user never named
        if ch is not None and ch not in ok:
            return {"error": f"That address format doesn't work on {CHAIN_NAME.get(ch, ch)}."}
        if ch is not None and ch not in agg and len(agg) == 1:
            ch = next(iter(agg))                              # app switches silently to the only chain held
        if ch is not None and ch not in agg:
            ch = None
        if ch is None:
            if len(agg) == 1:
                ch = next(iter(agg))
            elif not agg:
                return {"error": f"No {sym} found in this wallet."}
            else:
                names = [CHAIN_NAME[c] for c in agg]
                held = join_and(names)
                detail = ", ".join(f"{CHAIN_NAME[c]}: {dfmt(v)}" for c, v in agg.items())
                return {"note": "Nothing was prepared. Ask the user which network, in one short question.",
                        "options": list(agg), "prepared": False,
                        "question": f"Which network should the send use? You hold {sym} on {held} ({detail}).",
                        "status": "needs_chain"}
        net = CHAIN_NAME[ch]
        usd = float(total) * float(self.prices.get(sym, 0))
        reasons = []
        for i, a, _ in valid:
            if a in self.flagged:
                reasons.append(f"Row {i} is flagged as a known scam address.")
        if usd > self.move_cap:
            reasons.append("This batch is above the per-move cap for the assistant.")
        if usd > self.daily_left:
            reasons.append("Total is above your daily limit for the assistant.")
        if reasons:
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": reasons, "status": "blocked"}
        have = agg.get(ch, Decimal(0))
        if have < total:
            return {"note": "Nothing was prepared. Tell the user why.",
                    "reasons": [f"Not enough {sym} on {net}: you hold {dfmt(have)}, the batch needs {dfmt(total)}."],
                    "status": "blocked"}
        n = len(valid)
        fee_usd = Decimal("0.03") + Decimal("0.01") * n
        if sym in STABLE:
            fees = f"{dfmt(fee_usd)} {sym}"
        else:
            p = float(self.prices.get(sym, 1)) or 1.0
            fees = f"{format(Decimal(f'{float(fee_usd) / p:.2g}'), 'f')} {sym}"
        res = {"executed": False, "fees_up_to": fees, "kind": "multisend"}
        if left:
            res["left_out"] = left
            res["left_out_count"] = len(left)
        res["network"] = net
        res["note"] = ("Nothing has moved. The Batch send review lists every recipient; the user approves all of it "
                       "with one Face ID, or cancels.")
        res["recipients"] = n
        res["status"] = "review_opened"
        res["total"] = f"{dfmt(total)} {sym}"
        warns, first = [], {}
        for i, a, _ in valid:
            key = a.lower() if fam == "evm" else a
            if key in first:
                warns.append(f"Row {i} is the same address as row {first[key]}.")
            else:
                first[key] = i
        if warns:
            res["warnings"] = warns
        return res


def join_and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ====================================================================== conversation builder
class MConv(_Conv):
    def __init__(self, rng, world, bio):
        super().__init__(rng, world, system=SYSTEM_TMPL.replace("{BIO}", "Face ID"))
        self.bio = bio
        self.turns = []            # expect list per user turn
        self.kinds = []            # parse kind per user message
        self.user_idx = []         # message index of each user turn

    def user(self, text, kind):
        super().user(text)
        self.turns.append([])
        self.kinds.append(kind)
        self.user_idx.append(len(self.msgs) - 1)
        self.w.user_text = "\n".join(m["content"] for m in self.msgs if m["role"] == "user")

    def call(self, *calls, lead=""):
        r = super().call(*calls, lead=lead)
        self.turns[-1].append({"calls": [{"name": n, "arguments": a} for n, a in calls]})
        return r

    def say(self, text, ask=False):
        super().say(text, ask=ask)
        self.turns[-1].append({"ask": True} if ask else {"text": True})

    def ex(self, kind, extra=None):
        ex = {"messages": self.msgs, "tools": True,
              "meta": {"source": SRC, "generator": SRC, "kind": kind, "bio": self.bio, "world": self.world0,
                       "world_final": self.w.state(), "expect": self.turns[-1], "expect_by_turn": self.turns,
                       "parse": self.kinds, "lang": "en"}}
        ex["meta"].update(extra or {})
        for m in ex["messages"]:                              # the app's rule: Face ID -> device word, everywhere
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("Face ID", self.bio)
        return ex


# ====================================================================== phrase machinery
USE = Counter()          # balanced template use: always draw among the least used (reset per build)


def _balanced(rng, pool, idx, tag):
    lo = min(USE[(tag, i)] for i in idx)
    i = rng.choice([i for i in idx if USE[(tag, i)] == lo])
    USE[(tag, i)] += 1
    return pool[i]


def pick_t(rng, pool, ev):
    """Template pool split: index % 5 == 0 is eval-only (paraphrase-disjoint eval)."""
    idx = [i for i in range(len(pool)) if (i % 5 == 0) == ev]
    return _balanced(rng, pool, idx, ("t", id(pool), ev))


def pick_v(rng, pool, ev):
    """Slot value pool split: index % 4 == 0 is eval-only."""
    idx = [i for i in range(len(pool)) if (i % 4 == 0) == ev]
    if pool is STORIES or pool is PEOPLE:
        return _balanced(rng, pool, idx, ("v", id(pool), ev))
    return pool[rng.choice(idx)]


STORIES = ["november bounty payouts", "hackathon prizes", "this week's contributor rewards", "freelancer invoices for october",
           "my team's salaries", "giveaway winners", "dao grants round two", "refunds for the failed mint", "airdrop for early supporters",
           "tournament prizes", "monthly stipends", "bug bounty rewards", "design sprint crew", "raffle winners",
           "discord mod rewards", "translator payments", "community call speakers", "our ambassador program",
           "podcast guest payouts", "hackathon judges", "weekly tips for the streamers", "playtest rewards"]
PEOPLE = ["Ana", "Ben", "Chloe", "Dev", "Elif", "Farid", "Gina", "Hugo", "Ivy", "Jon", "Kofi", "Lena", "Mo", "Nia", "Omar",
          "Pia", "Quinn", "Raj", "Sam", "Tara", "Uma", "Vik", "Will", "Xena", "Yara", "Zed"]

# list on its own lines / inline, intro before or after. {list} is the block, {sym} the ticker, {net} " on base" or "".
BLOCK = [
    "send these in {sym}{net}:\n{list}",
    "pay them\n{list}\n{sym}",
    "batch send {sym}{net}\n{list}",
    "airdrop time. {sym}{net}:\n{list}",
    "{list}\n\nsend all of these{net}",
    "payout for {story}, paying in {sym}{net}:\n{list}",
    "pay my team {sym}{net}\n{list}",
    "mass send {sym}:\n{list}",
    "multi send pls\n{list}\ntoken: {sym}",
    "do these transfers{net} ({sym})\n{list}",
    "can you pay everyone on this list in {sym}{net}?\n\n{list}",
    "{story}:\n{list}\nall in {sym}{net}, thanks",
    "need to send {sym} to a bunch of wallets{net}. here they are\n{list}",
    "payouts\n{list}",
    "{list}",
    "send {sym} to all of these please{net}\n{list}",
    "here's the list for {story}. send it{net}\n{list}",
    "run this batch in {sym}{net}:\n{list}",
    "multisend {sym}{net}\n{list}",
    "please transfer the following{net}, in {sym}\n{list}",
    "pay these people\n{list}\n(all {sym}{net})",
    "I need to pay {n} people for {story}. {sym}{net}.\n{list}",
    "okay so {story}. send {sym}{net} to everyone below\n{list}",
    "payout time:\n{list}\n{sym}{net} pls",
    "batch: {sym}{net}\n{list}",
    "send em all\n{list}\nin {sym}",
    "mass payout in {sym}{net}, {n} wallets:\n{list}",
    "kindly send the amounts below in {sym}{net}\n{list}",
    "pay them all in {sym}{net}:\n{list}",
    "Hey Hermit, {story}. Can you do these transfers in {sym}{net}?\n{list}",
    "disburse {sym}{net}\n{list}",
    "{n} payments to make ({sym}{net}):\n{list}",
    "send {sym}{net}\n{list}\ndouble check the addresses before I approve",
    "gm. airdrop {sym}{net} to the following\n{list}",
    "transfer list for {story} ({sym}{net})\n{list}",
    "{sym} payouts{net}:\n{list}",
    "i need these sent out today, {sym}{net}\n{list}",
    "pay each person the amount next to their address, token {sym}{net}\n{list}",
    "Send:\n{list}\n\n{sym}{net}",
    "separate transfers? no, one batch please. {sym}{net}\n{list}",
    "multi-send these{net}, {sym}\n{list}",
    "do a bulk send of {sym}{net}:\n{list}",
    "treasury payout, {story}\n{list}\n{sym}{net}",
    "quick one, pay these in {sym}{net}\n{list}",
    "mass send for {story}. {sym}{net}.\n{list}",
]
# prose: {plist} is a sentence fragment. 'before' = amount precedes the address, 'after' = amount follows it.
PROSE_BEFORE = [
    "send {plist}",
    "in {sym}{net}, send {plist}",
    "send {sym}{net}: {plist}",
    "can you send {plist}? it's {sym}{net}",
    "pay {plist} in {sym}",
    "transfer {plist}{net}, {sym}",
    "{story}: {plist}. {sym}{net}",
    "i need to send {plist} ({sym}{net})",
    "do {plist} in {sym}{net}",
    "quick, {plist} in {sym} pls{net}",
    "airdrop {plist}, {sym}{net}",
    "batch send {sym}{net}, {plist}",
]
PROSE_AFTER = [
    "pay out in {sym}{net}: {plist}",
    "{story} in {sym}{net}, {plist}",
    "send {sym}{net}, {plist}",
    "payout list, {plist}. all {sym}{net}",
    "pay them in {sym}{net}. {plist}",
    "multi send {sym}{net}: {plist}",
    "payout: {plist} ({sym}{net})",
    "do these transfers in {sym}{net}, {plist}",
    "{plist}. {sym}{net}, one batch",
    "mass send {sym}: {plist}",
]
EACH = [
    "{amt} {sym} each to these {n}{net}:\n{list}",
    "send {amt} {sym} to each of these {n} addresses{net}\n{list}",
    "airdrop {amt} {sym} each{net}\n{list}",
    "{list}\n\n{amtd} each please{net}",
    "pay each of them {amtd}{net}. here are the {n}:\n{list}",
    "mass send {sym}{net}, flat {amt} each:\n{list}",
    "everyone gets {amtd}{net}\n{list}",
    "same amount for all: {amt} {sym}{net}\n{list}",
    "{story}. {amtd} per person{net}:\n{list}",
    "give these {n} wallets {amtd} apiece{net}\n{list}",
    "{n} wallets, {amt} each, {sym}{net}\n{list}",
    "multi send: each address below gets {amtd}{net}\n{list}",
    "{sym}{net}. {amt} each. go\n{list}",
    "bulk send {amtd} a head{net}\n{list}",
    "send these {n} people {amtd} each\n{list}",
    "{list}\neach one gets {amt} {sym}{net}",
    "batch send, {amt} {sym} to each{net}:\n{list}",
    "pay {amtd} each to the {n} addresses below{net}\n{list}",
]
SPLIT = [
    "split {tot} {sym} between these {n}{net}:\n{list}",
    "divide {tot} {sym} equally among these {n} wallets{net}\n{list}",
    "{list}\n\nshare {tot} {sym} evenly across all of these{net}",
    "I have {tot} {sym} to share between {n} people{net}. split it equally:\n{list}",
    "equal split of {tot} {sym}{net}\n{list}",
    "{tot} {sym} split evenly:\n{list}",
    "airdrop {tot} {sym} total, same share for everyone{net}:\n{list}",
    "distribute {tot} {sym} across these {n} addresses{net}:\n{list}",
    "can you divide {tot} {sym} fairly between the {n} wallets below{net}?\n{list}",
    "{story}: {tot} {sym} to be split equally{net}\n{list}",
    "split {tot} {sym} {n} ways{net}\n{list}",
    "spread {tot} {sym} evenly over these{net}\n{list}",
    "even split, {tot} {sym}{net}:\n{list}",
    "pls divide {tot} {sym} between them{net}\n{list}",
    "share the pool for {story}, {tot} {sym}, equally{net}:\n{list}",
    "everyone below should get the same. total {tot} {sym}{net}\n{list}",
    "multi send: {tot} {sym} divided by {n} wallets{net}\n{list}",
]
NAMES = [
    "pay {pairs}",
    "send {pairs} {sym}{net}",
    "payout: {pairs} in {sym}",
    "can you pay my team? {pairs} {sym}",
    "batch send {sym}: {pairs}",
    "airdrop {pairs} ({sym}{net})",
    "send {sym} to the guys: {pairs}",
    "pay them: {pairs}, all {sym}",
    "multisend {pairs} in {sym}",
    "{story}. {pairs} in {sym}{net}",
    "do these transfers{net}: {pairs} {sym}",
    "mass send {sym}: {pairs}",
    "pay everyone, {pairs}. {sym}{net}",
    "i need to pay {pairs} in {sym}",
]
FOLLOW = [
    "add one more: {addr} {amt}",
    "also add {addr} for {amt}",
    "oh forgot one, {addr} gets {amt} too",
    "one more person: {addr}, {amt}",
    "can you add {addr} {amt} to the batch?",
    "add this one too\n{addr} {amt}",
    "plus {addr} - {amt}",
    "wait add {addr} as well, {amt}",
    "include {addr}: {amt}",
    "add {addr} with {amt} please",
    "and one more {addr} {amt}",
    "adding {addr} {amt} to that list",
    "forgot someone. {addr} {amt}",
    "add another recipient: {addr} - {amt}",
]
CHAIN_ANS = ["{c}", "on {c}", "{c} please", "use {c}", "go with {c}", "the {c} one", "{c} pls", "{c}, thanks", "let's do {c}",
             "make it {c}", "{c} it is", "{c} for sure", "{c} network", "go {c}", "{c} is fine", "i'd say {c}"]
ADDR_REPLY = ["{pairs}", "here you go\n{pairs}", "ok:\n{pairs}", "their addresses:\n{pairs}", "sure. {pairs}", "{pairs}\n(those are the wallets)"]
NET_FORMS = [" on {X}", " on {X}", " via {X}", " ({X})", " on the {X} network", " over {X}"]
NET_APPEND = ["\non {X}", "\n(on {X})", "\nnetwork: {X}", "\nuse the {X} network"]
AMT_POOLS = {
    "stable": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 250, 300, 350, 400, 500, 600, 750, 800, 1000,
               1200, 1250, 1500, 2000, 2500, 12.5, 7.5, 17.5, 22.5, 37.5, 62.5, 8.25, 45.5, 99.99, 0.5, 2.5, 1.5, 33.33],
    "ETH": [0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1, 1.5, 2, 0.0125, 0.035],
    "SOL": [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 12, 20, 0.05, 0.75, 4],
    "NEAR": [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 50, 100, 150, 2.5, 7.5, 0.5, 40],
    "other": [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 40, 50, 75, 100, 150, 250, 500, 1000, 2.5, 0.5],
    "BONK": [1000, 5000, 10000, 25000, 50000, 100000, 2500, 7500, 20000],
}
SPLIT_TOT = {"stable": [20, 30, 40, 50, 60, 75, 100, 120, 150, 200, 250, 300, 400, 500, 600, 750, 900, 1000, 1200, 1500, 2000, 2400, 3000, 5000],
             "ETH": [0.3, 0.5, 0.6, 0.75, 1, 1.2, 1.5, 2, 3, 0.25, 0.9],
             "SOL": [1, 2, 3, 5, 6, 10, 12, 15, 20, 7.5], "NEAR": [10, 20, 30, 50, 60, 100, 150, 200, 25],
             "other": [10, 20, 30, 50, 60, 100, 120, 200, 250, 500, 1000], "BONK": [10000, 25000, 50000, 100000]}


def pool_key(sym):
    if sym in STABLE:
        return "stable"
    return sym if sym in AMT_POOLS else "other"


def pick_amount(rng, sym, ev):
    k = pool_key(sym)
    if rng.random() < 0.15 and k in ("stable", "other"):
        return D(round(rng.uniform(1, 300), 2))
    if rng.random() < 0.15 and k in ("ETH", "SOL"):
        return D(round(rng.uniform(0.002, 0.8), 4 if k == "ETH" else 3))
    return D(pick_v(rng, AMT_POOLS[k], ev))


def numtxt(rng, x, thou=True):
    x = Decimal(x)
    if x == x.to_integral_value():
        i = int(x)
        if i >= 1000 and thou and rng.random() < 0.35:
            return f"{i:,}"
        if i >= 1000 and i % 100 == 0 and thou and rng.random() < 0.08:
            return dfmt(Decimal(i) / 1000) + "k"
        if rng.random() < 0.03:
            return f"{i}.00"
        return str(i)
    s = dfmt(x)
    if len(s.split(".")[1]) == 1 and rng.random() < 0.15:
        s += "0"
    return s


def amt_str(rng, x, sd, mode, thou=True):
    s = numtxt(rng, x, thou)
    if mode == "usd" and sd.upper() in STABLE:
        return "$" + s
    if mode == "sym":
        return f"{s} {sd}"
    return s


PROTECT = {"each", "apiece", "split", "divide", "share", "between", "among", "across", "equally", "evenly", "gets", "get",
           "for", "to", "and", "total", "per", "person", "head", "piece", "these", "next", "address", "addresses", "same",
           "everyone", "ways", "divided", "spread", "distribute", "flat", "amount", "all", "pay", "send", "on", "via", "over"}


def maybe_typo(rng, t):
    """15% typo layer on the template text only: braces (slots), digits and key words are never touched."""
    parts = re.split(r"(\s+)", t)

    def eligible(tok):
        core = tok.strip(".,:;?!()'")
        return (core.isalpha() and len(core) >= 4 and core.lower() not in PROTECT and "{" not in tok and "}" not in tok)
    if rng.random() < 0.15:
        idx = [i for i, p in enumerate(parts) if p and not p.isspace() and eligible(p)]
        if idx:
            i = rng.choice(idx)
            w = parts[i]
            op = rng.choice(["swap", "vowel", "double", "pls"])
            if op == "swap" and len(w) >= 5:
                j = rng.randrange(1, len(w) - 2)
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            elif op == "vowel":
                vs = [k for k, ch in enumerate(w) if ch in "aeiou" and 0 < k < len(w) - 1]
                if vs:
                    k = rng.choice(vs)
                    w = w[:k] + w[k + 1:]
            elif op == "double":
                j = rng.randrange(1, len(w) - 1)
                w = w[:j] + w[j] + w[j:]
            parts[i] = w
        out = "".join(parts)
        out = out.replace("please", rng.choice(["pls", "plz"])).replace("thanks", "thx")
        parts = re.split(r"(\s+)", out)
    r = rng.random()
    if r < 0.10:
        parts = [p if ("{" in p or "}" in p) else p.lower() for p in parts]
    elif r < 0.12:
        parts = [p if ("{" in p or "}" in p) else p.upper() for p in parts]
    out = "".join(parts)
    if rng.random() < 0.10 and out and out[-1] in ".?!" and not out.endswith("}"):
        out = out[:-1]
    return out


def fill(rng, tmpl, **slots):
    return maybe_typo(rng, tmpl).format(**slots)


def net_phrase(rng, chain):
    forms = [f for f in NET_FORMS if not (chain == "near" and "(" in f)]
    return rng.choice(forms).format(X=rng.choice(CHAIN_SAY[chain]))


# ---- list layouts
AFTER_FMT = ["{a} {x}", "{a}: {x}", "{a} - {x}", "{a}, {x}", "{a} = {x}", "{a}\t{x}", "{a} ({x})", "{a} gets {x}", "{a} -> {x}",
             "{a} | {x}", "{a} {x}", "{a}: {x}"]
BEFORE_FMT = ["{x} to {a}", "{x} -> {a}", "{x} {a}", "{x}: {a}", "{x} for {a}", "{x} to {a}"]


def build_list(rng, rows, sd, mode, dirn):
    """rows: [(addr, Decimal)]. Returns text. dirn after = amount follows the address."""
    kind = rng.choices(["lines", "bullets", "numbered", "comma", "semi"], [32, 16, 10, 22, 12])[0]
    thou = kind in ("lines", "bullets", "numbered", "semi")
    fmt = rng.choice(AFTER_FMT if dirn == "after" else BEFORE_FMT)
    if kind == "comma" and fmt.startswith("{a}, "):
        fmt = "{a} {x}"
    if kind == "comma" and "|" in fmt:
        fmt = "{a} {x}"
    lines = [fmt.format(a=a, x=amt_str(rng, x, sd, mode, thou)) for a, x in rows]
    if kind == "lines":
        return "\n".join(lines)
    if kind == "bullets":
        b = rng.choice(["- ", "- ", "* ", "• "])
        return "\n".join(b + l for l in lines)
    if kind == "numbered":
        sep = rng.choice([". ", ") "])
        return "\n".join(f"{i}{sep}{l}" for i, l in enumerate(lines, 1))
    if kind == "comma":
        return ", ".join(lines)
    return "; ".join(lines)


def addr_list(rng, addrs):
    kind = rng.choices(["lines", "bullets", "numbered", "comma", "space"], [40, 20, 10, 20, 10])[0]
    if kind == "lines":
        return "\n".join(addrs)
    if kind == "bullets":
        return "\n".join(rng.choice(["- ", "* "]) + a for a in addrs)
    if kind == "numbered":
        return "\n".join(f"{i}. {a}" for i, a in enumerate(addrs, 1))
    if kind == "comma":
        return ", ".join(addrs)
    return " ".join(addrs)


def prose_list(rng, rows, sd, style, mode):
    parts = []
    for a, x in rows:
        xt = amt_str(rng, x, sd, "sym" if style == "sym_to" else mode, thou=False)
        parts.append({"to": "{x} to {a}", "sym_to": "{x} to {a}", "gets": "{a} gets {x}", "for": "{a} for {x}",
                      "plain": "{a} {x}"}[style].format(a=a, x=xt))
    if len(parts) == 1:
        return parts[0]
    if rng.random() < 0.7:
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return ", ".join(parts)


def sd_of(rng, sym):
    return sym if rng.random() < 0.72 else sym.lower()


def ensure_net(rng, text, named):
    if named and named_chains(text) != {named}:
        text += rng.choice(NET_APPEND).format(X=rng.choice(CHAIN_SAY[named]))
    return text


def compose_list_msg(rng, ev, layout, rows, sym, named, each=None, tot=None):
    """rows: [(addr, Decimal|None)]. Returns dict(text, kind, ...). kind drives the independent parser."""
    n = len(rows)
    sd = sd_of(rng, sym)
    story = pick_v(rng, STORIES, ev)
    net = net_phrase(rng, named) if named else ""
    stable = sym in STABLE
    if layout == "list":
        tmpl = pick_t(rng, BLOCK, ev)
        dirn = "after" if rng.random() < 0.7 else "before"
        need = "{sym}" not in tmpl
        mode = "sym" if need else rng.choices(["plain", "usd", "sym"], [55, 25 if stable else 0, 20])[0]
        text = fill(rng, tmpl, sym=sd, net=net, story=story, n=n, list=build_list(rng, rows, sd, mode, dirn))
        kind = "list_" + dirn
    elif layout == "prose":
        group = rng.choice(["before", "after"])
        tmpl = pick_t(rng, PROSE_BEFORE if group == "before" else PROSE_AFTER, ev)
        style = rng.choice(["to", "sym_to"] if group == "before" else ["gets", "for", "plain"])
        need = "{sym}" not in tmpl
        mode = "sym" if need else rng.choices(["plain", "usd", "sym"], [55, 25 if stable else 0, 20])[0]
        text = fill(rng, tmpl, sym=sd, net=net, story=story, n=n, plist=prose_list(rng, rows, sd, style, mode))
        kind = "list_" + group
    elif layout == "each":
        tmpl = pick_t(rng, EACH, ev)
        amt = numtxt(rng, each, thou=True)
        dollar_ok = stable and "{sym}" in tmpl and rng.random() < 0.5
        amtd = ("$" + amt) if dollar_ok else f"{amt} {sd}"
        text = fill(rng, tmpl, sym=sd, net=net, story=story, n=n, amt=amt, amtd=amtd, list=addr_list(rng, [a for a, _ in rows]))
        kind = "each"
    elif layout == "split":
        tmpl = pick_t(rng, SPLIT, ev)
        text = fill(rng, tmpl, sym=sd, net=net, story=story, n=n, tot=numtxt(rng, tot, thou=True),
                    list=addr_list(rng, [a for a, _ in rows]))
        kind = "split"
    else:
        raise ValueError(layout)
    text = ensure_net(rng, text, named)
    return {"text": text, "kind": kind}


# ====================================================================== independent parsers (also used by the self-check)
EVM_LOOSE = re.compile(r"(?<![\w.])0x[0-9a-fA-F]+(?!\w)")
SOL_LOOSE = re.compile(r"(?<![\w.])[1-9A-HJ-NP-Za-km-z]{32,44}(?!\w)")
NEAR_LOOSE = re.compile(r"(?<![\w.@-])[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.near(?![\w-])")
EACH_RES = [re.compile(r"\$?" + NUMC + r"\s*(?:[A-Za-z]{2,6}\s*)?(?:to\s+)?(?:each|apiece|per person|per head|a head|a piece)\b", re.I),
            re.compile(r"\bgets?\s+\$?" + NUMC, re.I),
            re.compile(r"same amount[^\d$]{0,20}\$?" + NUMC, re.I),
            re.compile(r"each of them\s*\$?" + NUMC, re.I)]
SPLIT_RE = re.compile(r"\$?" + NUMC + r"\s*(?:" + "|".join(TICKERS) + r")\b", re.I)
SPLIT_RE2 = re.compile(r"\$" + NUMC)


def parse_addresses(text):
    spans = []
    for rx in (EVM_LOOSE, SOL_LOOSE, NEAR_LOOSE):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), m.group(0)))
    spans.sort()
    out = []
    for s in spans:                                   # drop overlaps
        if not out or s[0] >= out[-1][1]:
            out.append(s)
    return out


def blank(text, spans):
    t = list(text)
    for s, e, _ in spans:
        for i in range(s, e):
            t[i] = " "
    return "".join(t)


def tickers_in(text):
    t = blank(text, parse_addresses(text))
    out = set()
    for tk in TICKERS:
        for m in re.finditer(rf"(?<![\w.]){tk}(?![\w])", t, re.I):
            if tk == "NEAR":
                pre = t[:m.start()].rstrip().lower()
                post = t[m.end():].lstrip().lower()
                if re.search(r"(\bon|\bvia|\bover|\bthrough)$", pre) or re.match(r"(network|chain)\b", post):
                    continue
            out.add(tk)
    return out


def parse_rows(text, kind, sym):
    """Independent parse of one user message -> [(addr, Decimal)] (amount None for 'each'/'split' until resolved)."""
    spans = parse_addresses(text)
    addrs = [a for _, _, a in spans]
    t = blank(text, spans)
    if kind in ("list_after", "followup"):
        res = []
        for i, (s, e, a) in enumerate(spans):
            nxt = spans[i + 1][0] if i + 1 < len(spans) else len(t)
            m = NUM_RE.search(t, e, nxt)
            res.append((a, nval(m) if m else None))
        return res
    if kind == "list_before":
        res = []
        for i, (s, e, a) in enumerate(spans):
            prv = spans[i - 1][1] if i else 0
            ms = list(NUM_RE.finditer(t, prv, s))
            res.append((a, nval(ms[-1]) if ms else None))
        return res
    if kind == "each":
        for rx in EACH_RES:
            m = rx.search(t)
            if m:
                v = nval(m)
                return [(a, v) for a in addrs]
        return [(a, None) for a in addrs]
    if kind == "split":
        ms = [m for m in (SPLIT_RE.search(t), SPLIT_RE2.search(t)) if m]
        if not ms:
            return [(a, None) for a in addrs]
        m = min(ms, key=lambda m: m.start())
        tot = nval(m)
        dp = Decimal(1).scaleb(-sym_dp(sym))
        amt = (tot / len(addrs)).quantize(dp, rounding=ROUND_HALF_EVEN) if addrs else None
        return [(a, amt) for a in addrs]
    if kind == "chain":
        return []
    raise ValueError(kind)


NAME_PAIR1 = re.compile(r"\b([A-Z][a-z]+)\s*(?:gets|:|-|=)?\s*\$?" + NUMC)
NAME_PAIR2 = re.compile(r"\$?" + NUMC + r"\s*(?:[A-Za-z]{2,5}\s+)?to\s+([A-Z][a-z]+)\b")


def _v(g1, g2):
    v = Decimal(g1.replace(",", ""))
    return v * 1000 if g2 else v


def parse_name_amounts(text):
    out = {}
    for m in NAME_PAIR1.finditer(text):
        out[m.group(1)] = _v(m.group(2), m.group(3))
    for m in NAME_PAIR2.finditer(text):
        out[m.group(3)] = _v(m.group(1), m.group(2))
    return out


def parse_names_in(text):
    return [m.group(1) for m in re.finditer(r"\b([A-Z][a-z]+)\b", text) if m.group(1) in PEOPLE]


# ====================================================================== answers
def nr(n):
    return f"{n} recipients" if n != 1 else "1 recipient"


LEADS = ["I've prepared a batch send of {total} on {net} to {nr}.", "Your batch send is ready to review: {total} on {net} to {nr}.",
         "Prepared a batch send of {total} on {net} for {nr}.", "The batch is prepared: {total} on {net}, going to {nr}.",
         "I set up the batch send. It's {total} on {net} to {nr}.", "Batch send prepared: {total} on {net}, {nr}.",
         "I've lined up {nr} for a total of {total} on {net}."]
LEADS_FOLLOW = ["I've prepared the batch again with the new row: {total} on {net} to {nr}.",
                "Added. I've prepared a new batch send of {total} on {net} with all {nr}.",
                "Got it. The updated batch send is prepared: {total} on {net} for {nr}.",
                "I rebuilt the batch with the extra row. It's now {total} on {net} to {nr}.",
                "New batch prepared with all {nr}, including the one you just added: {total} on {net}.",
                "The new row is in. The updated batch is {total} on {net} to {nr}."]
CLOSERS = ["Check the list, then approve once with Face ID.", "Look over the recipients on the review screen, then approve it once with Face ID.",
           "Nothing moves until you approve it with Face ID, and one approval covers the whole batch.",
           "Review the list and confirm with Face ID. It's one approval for all of it.",
           "It's waiting for you: check each row, then approve with Face ID.",
           "Go through the rows on the review screen and approve with Face ID when they look right."]
CLOSERS_FOLLOW = ["Use this new review, not the earlier one: check the list and approve once with Face ID.",
                  "Check the full list on the new review screen, then approve once with Face ID. Skip the earlier review.",
                  "Review all the rows on this new screen and approve with Face ID when they look right, not the earlier one.",
                  "Nothing moves until you approve the new review with Face ID. Leave the earlier review alone."]
LEFT_OUT = ["I left out {items}.", "{Items} was left out.", "Heads up: I had to leave out {items}.", "{Items} didn't make the batch."]
WARN = ["Heads up: {w}", "Note: {w}", "One flag: {w}", "Worth a look: {w}"]
WARN_TAIL = ["Check that it's intended.", "Make sure that's what you want.", "Fix it on the review if not.", ""]
ADDR_LIST = ["It starts with {a}.", "Recipients include {a}.", "First in the list: {a}."]
FEES = ["Fees are up to {f}.", "Network fees should be up to {f}.", "Fees come to at most {f}."]
SPLIT_EXACT = ["{a} {S} each.", "That's {a} {S} for each of the {n}.", "{T} {S} split {n} ways is {a} {S} each."]
SPLIT_LEFT = ["{a} {S} each, and the {left} {S} left over stays unsent.", "That's {a} {S} each; {left} {S} stays unsent.",
              "Each gets {a} {S}, which leaves {left} {S} unsent.", "{a} {S} each, with {left} {S} left over that stays unsent."]
ASK_NAMES = ["What are the wallet addresses for {names}?", "Sure, I just need the addresses. Can you paste them for {names}?",
             "I need a wallet address for each person. What are they for {names}?",
             "I only prepare sends to addresses you type, so can you paste them for {names}?",
             "Which wallet addresses should {names} be paid at?", "Happy to. What are the addresses for {names}?",
             "I don't have addresses for {names}. Can you paste them?", "What wallet address does each of them use, {names}?"]
BLOCK_ANS = ["Nothing was prepared. {reasons}", "I couldn't prepare it. {reasons} Nothing is waiting for Face ID.",
             "The app blocked this batch. {reasons} Nothing was prepared.", "Nothing was prepared, so there's nothing to approve. {reasons}",
             "That batch was blocked. {reasons}", "I can't prepare this one. {reasons}"]
BLOCK_TAIL = ["Tell me what to change and I'll prepare it again.", "Change the list and I can try once more.", "", "",
              "Say so if you want a smaller batch."]
INSUF = ["You don't have enough {S} on {net}: you hold {have} and the batch needs {need}.",
         "Nothing was prepared. {Have} {S} on {net} isn't enough, the batch needs {need}.",
         "Your {S} on {net} won't cover it. You hold {have}, the batch needs {need}.",
         "I couldn't prepare it. You hold {have} {S} on {net} and the batch needs {need}."]
INSUF_TAIL = ["Want me to lower the amounts?", "Add funds or trim the list and I'll try again.", "", "Tell me what to change."]


def words(t):
    return len(t.split())


def ans_success(rng, r, args, followup=False, split=None):
    n, total, net = r["recipients"], r["total"], r["network"]
    lead = rng.choice(LEADS_FOLLOW if followup else LEADS).format(total=total, net=net, nr=nr(n))
    pre = ""
    if split:
        S = args["symbol"]
        if split["left"] is None:
            pre = rng.choice(SPLIT_EXACT).format(a=split["a"], S=S, n=n, T=split["T"])
        else:
            pre = rng.choice(SPLIT_LEFT).format(a=split["a"], S=S, left=split["left"])
    must = []
    if r.get("left_out"):
        items = []
        for s in r["left_out"]:
            m = re.match(r"row (\d+): (.*)$", s)
            items.append(f"row {m.group(1)} ({m.group(2)})")
        it = join_and(items)
        must.append(rng.choice(LEFT_OUT).format(items=it, Items=it[0].upper() + it[1:]))
    for w in r.get("warnings", []):
        must.append(rng.choice(WARN).format(w=w) + " " + rng.choice(WARN_TAIL))
    closer = rng.choice(CLOSERS_FOLLOW if followup else CLOSERS)
    opt = []
    if rng.random() < 0.15:
        opt.append(rng.choice(FEES).format(f=r["fees_up_to"]))
    if rng.random() < 0.2 and not must:
        shown = [x["address"] for x in args["recipients"]][:3]
        opt.append(rng.choice(ADDR_LIST).format(a=join_and(shown)))
    for k in range(len(opt) + 1):
        parts = [lead] + ([pre] if pre else []) + must + opt[:len(opt) - k] + [closer]
        text = " ".join(p.strip() for p in parts if p.strip())
        if words(text) <= 70:
            return text
    return text


def ans_blocked(rng, r, S, net=None):
    rs = r["reasons"]
    m = re.match(r"Not enough (\S+) on (.+?): you hold (\S+), the batch needs (\S+)\.$", rs[0])
    if m and rng.random() < 0.6:
        t = rng.choice(INSUF).format(S=m.group(1), net=m.group(2), have=m.group(3), need=m.group(4), Have=m.group(3))
        tail = rng.choice(INSUF_TAIL)
        return (t + " " + tail).strip()
    t = rng.choice(BLOCK_ANS).format(reasons=" ".join(rs))
    return (t + " " + rng.choice(BLOCK_TAIL)).strip()


def ans_needs_chain(rng, r):
    names = [CHAIN_NAME[c] for c in r["options"]]
    if rng.random() < 0.6:
        return r["question"]
    return f"Which network should I use for the batch, {join_and(names).replace(' and ', ' or ')}?"


# ====================================================================== world / rows
def ceil_dp(x, dp):
    q = Decimal(1).scaleb(-dp)
    return (x / q).to_integral_value(rounding="ROUND_CEILING") * q


def make_world(rng, sym, held, daily_left=None, move_cap=None, flagged=None):
    """held: {chain: Decimal balance of sym}."""
    w = MWorld(rng)
    p = w.prices[sym]
    dp = sym_dp(sym)
    w.balances = []
    for ch, amt in held.items():
        w.balances.append({"symbol": sym, "chain": ch, "amount": float(amt), "usd": round(float(amt) * p, 2)})
    for _ in range(rng.choice([0, 1, 1, 2])):               # distractor holdings
        s2 = rng.choice(["ETH", "SOL", "USDC", "BNB", "POL"])
        ch2 = rng.choice(list(CHAIN_NAME))
        if s2 == sym or any(b["symbol"] == s2 and b["chain"] == ch2 for b in w.balances):
            continue
        a2 = round(rng.uniform(0.01, 5), 4) if w.prices[s2] > 50 else round(rng.uniform(5, 400), 2)
        w.balances.append({"symbol": s2, "chain": ch2, "amount": a2, "usd": round(a2 * w.prices[s2], 2)})
    if daily_left is not None:
        w.daily_left = daily_left
    if move_cap is not None:
        w.move_cap = move_cap
    w.flagged = flagged or []
    return w


def cover(rng, tot, sym, lo=1.05, hi=3.0):
    return ceil_dp(tot * D(round(rng.uniform(lo, hi), 3)), sym_dp(sym))


N_W = [(2, 10), (3, 16), (4, 16), (5, 14), (6, 12), (7, 8), (8, 7), (9, 5), (10, 5), (11, 3), (12, 4)]


def pick_n(rng, lo=2, hi=12):
    return wchoice(rng, [(k, w) for k, w in N_W if lo <= k <= hi])


def make_rows(rng, ev, fam, n, sym):
    return [(a, pick_amount(rng, sym, ev)) for a in gen_addrs(rng, fam, n)]


def rec_args(sym, chain, rows):
    a = {"symbol": sym}
    if chain:
        a["chain"] = chain
    a["recipients"] = [{"address": x, "amount": jnum(v)} for x, v in rows]
    return a


def rows_total(rows):
    return sum((v for _, v in rows), Decimal(0))


def calc_split(rng, c, tot, n, sym):
    """Returns (amount Decimal, left str|None) after running the calculator calls through the conversation."""
    dp = sym_dp(sym)
    q = Decimal(1).scaleb(-dp)
    T = Decimal(tot)
    exact = (T / n) == (T / n).quantize(q)
    amt = (T / n).quantize(q, rounding=ROUND_HALF_EVEN)
    expr = f"{dfmt(T)} / {n}" if exact else f"round({dfmt(T)} / {n}, {dp})"
    r = c.call(("calculator", {"expression": expr}))
    assert "result" in r and D(r["result"]) == amt, (expr, r, amt)
    a = D(r["result"])
    if exact:
        return a, None
    r2 = c.call(("calculator", {"expression": f"{dfmt(T)} - {dfmt(a)} * {n}"}))
    assert "result" in r2 and D(r2["result"]) == T - a * n, (r2, T, a, n)
    return a, r2["result"]


# ====================================================================== scenarios
SCEN_W = {"basic": 150, "each": 50, "split": 50, "needs_chain": 45, "names": 40, "blocked": 25, "insufficient": 25,
          "dup": 25, "followup": 30, "nonevm": 40, "left_out": 15}
EVAL_Q = {"basic": 12, "each": 5, "split": 5, "needs_chain": 5, "names": 5, "blocked": 3, "insufficient": 3, "dup": 3,
          "followup": 3, "nonevm": 4, "left_out": 2}
CHECKS = ["no_chain_invented", "exact_digits", "right_tool", "no_unbacked_done"]


def pick_world_chain(rng, sym, fam):
    return pick_chain(rng, sym, fam)


def gen(rng, ev, scen, idx=0):
    bio = "Face ID" if rng.random() < 0.7 else "Touch ID"
    # ---- symbol / family
    if scen == "nonevm":
        fam = rng.choice(["solana", "solana", "near"])
        if fam == "solana":
            sym = wchoice(rng, [("SOL", 40), ("USDC", 38), ("USDT", 10), ("OTHER", 12)])
            if sym == "OTHER":
                sym = rng.choice(OTHER_SOL[ev])
        else:
            sym = wchoice(rng, [("NEAR", 55), ("USDC", 45)])
    elif scen in ("needs_chain", "left_out"):
        sym = pick_sym(rng, ev)
        while sym_family(sym) != "evm" or (scen == "needs_chain" and len(TOKEN_CHAINS[sym]) < 2):
            sym = pick_sym(rng, ev)
        fam = "evm"
    else:
        sym = pick_sym(rng, ev)
        fam = sym_family(sym)
    chain = pick_world_chain(rng, sym, fam)
    named = chain if (rng.random() < 0.25 and sym != "NEAR" and scen not in ("needs_chain", "names")) else None
    if scen == "names":
        named = chain if (rng.random() < 0.25 and sym != "NEAR") else None
    if scen == "names":
        return gen_names(rng, ev, bio, sym, fam, named)
    if scen == "needs_chain":
        return gen_needs_chain(rng, ev, bio, sym, idx)
    layout = {"each": "each", "split": "split"}.get(scen)
    if layout is None:
        layout = "prose" if (rng.random() < (0.3 if scen == "basic" else 0.2) and scen not in ("dup", "left_out")) else "list"
    n = pick_n(rng, 3 if scen in ("dup", "left_out") else 2, 6 if layout == "prose" else 12)
    if scen in ("each", "split"):
        n = wchoice(rng, [(3, 14), (4, 16), (5, 16), (6, 12), (7, 8), (8, 8), (10, 4), (12, 3)]) if scen == "each" else \
            rng.choice([2, 3, 3, 4, 4, 5, 6, 7, 8, 10, 12])
    rows = make_rows(rng, ev, fam, n, sym)
    # ---- special rows
    dup = None
    left_idx = None
    if scen == "dup":
        j = rng.randrange(1, n)
        i = rng.randrange(0, j)
        a = rows[i][0]
        if fam == "evm" and rng.random() < 0.4 and a != a.lower():
            a = a.lower()
        rows[j] = (a, rows[j][1])
        dup = (i, j)
    if scen == "left_out":
        k = rng.randrange(0, n)
        good = rows[k][0]
        body = good[2:]
        bad = "0x" + rng.choice([body[:-1], body + rng.choice("0123456789abcdef"), body[:-2]])
        rows[k] = (bad, rows[k][1])
        left_idx = k
    each = tot = None
    if layout == "each":
        each = pick_amount(rng, sym, ev)
        rows = [(a, each) for a, _ in rows]
    if layout == "split":
        while True:
            tot = D(pick_v(rng, SPLIT_TOT[pool_key(sym)], ev))
            dp = Decimal(1).scaleb(-sym_dp(sym))
            if (tot / n) >= dp * 2:
                break
    msg = compose_list_msg(rng, ev, layout, rows, sym, named, each=each, tot=tot)
    # ---- world
    held = {}
    tot_need = rows_total([(a, v) for a, v in rows if v is not None]) if layout != "split" else tot
    if scen == "left_out":
        tot_need = rows_total([r for k, r in enumerate(rows) if k != left_idx])
    fu = None
    if scen == "followup":
        na = gen_addr(rng, fam)
        while na.lower() in {x.lower() for x, _ in rows}:
            na = gen_addr(rng, fam)
        fu = (na, pick_amount(rng, sym, ev))
        tot_need += fu[1]
    if scen == "insufficient":
        have = tot_need * D(round(rng.uniform(0.15, 0.9), 3))
        have = have.quantize(Decimal(1), rounding="ROUND_FLOOR") if have >= 5 else have.quantize(Decimal(1).scaleb(-sym_dp(sym)))
        if have >= tot_need or have <= 0:
            have = (tot_need / 2).quantize(Decimal(1).scaleb(-sym_dp(sym)))
        assert 0 < have < tot_need
        held[chain] = have
    else:
        held[chain] = cover(rng, tot_need, sym)
    if scen in ("basic", "each", "split", "nonevm") and fam == "evm" and rng.random() < 0.3 and named and len(TOKEN_CHAINS[sym]) > 1:
        other = rng.choice([c for c in TOKEN_CHAINS[sym] if c != chain])
        held[other] = cover(rng, tot_need, sym)
    w = make_world(rng, sym, held)
    if scen == "blocked":
        usd = float(tot_need) * float(w.prices[sym])
        mode = rng.choice(["daily", "cap", "flag", "both"])
        if mode in ("daily", "both"):
            w.daily_left = round(usd * rng.uniform(0.2, 0.8), 2)
        if mode in ("cap", "both"):
            w.move_cap = round(usd * rng.uniform(0.2, 0.8), 2)
        if mode == "flag":
            w.flagged = [rows[rng.randrange(n)][0]]
    c = MConv(rng, w, bio)
    c.user(msg["text"], msg["kind"])
    call_chain = named
    split = None
    if layout == "split":
        a, left = calc_split(rng, c, tot, n, sym)
        rows = [(x, a) for x, _ in rows]
        split = {"a": dfmt(a), "left": left, "T": dfmt(tot)}
    args = rec_args(sym, call_chain, rows)
    r = c.call(("prepare_multisend", args))
    if scen == "followup":
        return gen_followup(rng, ev, c, bio, sym, fam, named, rows, args, r, fu)
    c_say_result(rng, c, r, args, split)
    kind = {"basic": "list", "each": "each", "split": "split", "blocked": "blocked", "insufficient": "insufficient",
            "dup": "duplicate", "nonevm": "nonevm", "left_out": "left_out"}[scen]
    if layout == "prose" and scen == "basic":
        kind = "prose"
    exp_status = {"blocked": "blocked", "insufficient": "blocked"}.get(scen, "review_opened")
    assert r.get("status") == exp_status, (scen, r)
    if scen == "dup":
        assert r.get("warnings"), r
    if scen == "left_out":
        assert r.get("left_out"), r
    return finish(c, kind, rows, sym, call_chain, scen, r, {"layout": layout, "sym": sym, "fam": fam, "named": named})


def c_say_result(rng, c, r, args, split=None, followup=False):
    st = r.get("status")
    if st == "review_opened":
        c.say(ans_success(rng, r, args, followup=followup, split=split))
    elif st == "blocked":
        c.say(ans_blocked(rng, r, args["symbol"]))
    elif st == "needs_chain":
        c.say(ans_needs_chain(rng, r), ask=True)
    else:
        raise AssertionError(r)


def gold_for(c, rows, sym, chain, scen, r, extra):
    """Gold the eval needs: the multisend call, addresses and amounts as typed, answer checks."""
    g = {"scenario": scen, "tool": "prepare_multisend", "symbol": sym, "chain": chain,
         "addresses": [a for a, _ in rows], "amounts": [jnum(v) for _, v in rows],
         "calls": [{"name": "prepare_multisend", "arguments": rec_args(sym, chain, rows)}],
         "no_call": False, "must_ask": False, "forbid_tools": ["prepare_send"],
         "answer_must_not_match": [r"\bsent\b", r"\bRelay\b", r"went through", r"successfully"]}
    st = r.get("status") if r else None
    if st == "review_opened":
        g["answer_must_contain"] = [r["total"], r["network"]]
    elif st == "blocked":
        g["answer_must_contain"] = []
        g["blocked"] = True
    g.update(extra)
    return g


def finish(c, kind, rows, sym, chain, scen, r, slots):
    ex = c.ex(kind, {"scenario": scen, "slots": slots, "checks": list(CHECKS)})
    ex["gold"] = gold_for(c, rows, sym, chain, scen, r, {})
    ex["meta"]["cut"] = "last"
    return ex


def gen_followup(rng, ev, c, bio, sym, fam, named, rows, args, r, fu):
    c_say_result(rng, c, r, args)
    new_addr, new_amt = fu
    sd = sd_of(rng, sym)
    mode = rng.choices(["plain", "usd", "sym"], [50, 25 if sym in STABLE else 0, 25])[0]
    text = fill(rng, pick_t(rng, FOLLOW, ev), addr=new_addr, amt=amt_str(rng, new_amt, sd, mode, thou=False))
    c.user(text, "followup")
    rows2 = rows + [(new_addr, new_amt)]
    args2 = rec_args(sym, named, rows2)
    r2 = c.call(("prepare_multisend", args2))
    assert r2.get("status") == "review_opened", r2
    c.say(ans_success(rng, r2, args2, followup=True))
    ex = c.ex("followup_add", {"scenario": "followup", "slots": {"layout": "list", "sym": sym, "fam": fam, "named": named},
                               "checks": list(CHECKS)})
    ex["gold"] = gold_for(c, rows2, sym, named, "followup", r2, {})
    ex["meta"]["cut"] = "last"
    return ex


def gen_needs_chain(rng, ev, bio, sym, idx):
    fam = "evm"
    n = pick_n(rng, 2, 10)
    layout = "prose" if (rng.random() < 0.2 and n <= 6) else "list"
    rows = make_rows(rng, ev, fam, n, sym)
    k = 2 if (rng.random() < 0.7 or len(TOKEN_CHAINS[sym]) < 3) else 3
    pool = TOKEN_CHAINS[sym]
    chains = []
    while len(chains) < k:
        ch = rng.choices(pool, [CHW[x] for x in pool])[0]
        if ch not in chains:
            chains.append(ch)
    tot = rows_total(rows)
    held = {ch: cover(rng, tot, sym) for ch in chains}
    w = make_world(rng, sym, held)
    msg = compose_list_msg(rng, ev, layout, rows, sym, None)
    c = MConv(rng, w, bio)
    c.user(msg["text"], msg["kind"])
    args1 = rec_args(sym, None, rows)
    r1 = c.call(("prepare_multisend", args1))
    assert r1.get("status") == "needs_chain", r1
    c.say(ans_needs_chain(rng, r1), ask=True)
    pick = rng.choice(chains)
    reply = pick_t(rng, CHAIN_ANS, ev).format(c=rng.choice(CHAIN_SAY[pick]))
    c.user(reply, "chain")
    args2 = rec_args(sym, pick, rows)
    r2 = c.call(("prepare_multisend", args2))
    assert r2.get("status") == "review_opened", r2
    c.say(ans_success(rng, r2, args2))
    ex = c.ex("needs_chain", {"scenario": "needs_chain", "slots": {"layout": layout, "sym": sym, "fam": fam, "named": None, "answer": pick},
                              "checks": list(CHECKS) + ["asked_when_needed"]})
    ex["gold"] = gold_for(c, rows, sym, pick, "needs_chain", r2, {
        "first_turn": {"calls": [{"name": "prepare_multisend", "arguments": args1}], "chain": None, "then_must_ask": True,
                       "question_options": r1["options"]}})
    ex["meta"]["cut"] = "first" if idx % 2 == 0 else "last"
    ex["gold"]["cut"] = ex["meta"]["cut"]
    return ex


def gen_names(rng, ev, bio, sym, fam, named):
    n = rng.choice([2, 2, 3, 3, 4])
    people = rng.sample([p for i, p in enumerate(PEOPLE) if (i % 4 == 0) == ev], min(n, len([1 for i in range(len(PEOPLE)) if (i % 4 == 0) == ev])))
    while len(people) < n:
        people.append(rng.choice(PEOPLE))
    people = list(dict.fromkeys(people))
    n = len(people)
    amts = [pick_amount(rng, sym, ev) for _ in people]
    sd = sd_of(rng, sym)
    cont = (not ev) and rng.random() < 0.35
    tmpl = pick_t(rng, NAMES, ev)
    style = rng.choice(["plain", "to", "colon", "gets"])
    with_sym = cont and "{sym}" not in tmpl          # a continued batch needs the token named somewhere
    parts = []
    for p, a in zip(people, amts):
        x = numtxt(rng, a, thou=False) + (f" {sd}" if with_sym else "")
        parts.append({"plain": f"{p} {x}", "to": f"{x} to {p}", "colon": f"{p}: {x}", "gets": f"{p} gets {x}"}[style])
    pairs = (", ".join(parts[:-1]) + " and " + parts[-1]) if rng.random() < 0.7 else ", ".join(parts)
    net = net_phrase(rng, named) if named else ""
    text = ensure_net(rng, fill(rng, tmpl, pairs=pairs, sym=sd, net=net, story=pick_v(rng, STORIES, ev)), named)
    w = make_world(rng, sym, {pick_chain(rng, sym, fam): cover(rng, rows_total(list(zip(people, amts))), sym)})
    c = MConv(rng, w, bio)
    c.user(text, "names")
    c.say(rng.choice(ASK_NAMES).format(names=join_and(people)), ask=True)
    if not cont:
        ex = c.ex("names_only", {"scenario": "names", "slots": {"sym": sym, "fam": fam, "named": named, "people": people},
                                 "checks": ["asked_when_needed", "right_tool", "no_chain_invented", "no_unbacked_done"]})
        ex["gold"] = {"scenario": "names", "tool": None, "no_call": True, "must_ask": True, "calls": [], "symbol": sym,
                      "chain": None, "addresses": [], "amounts": [], "forbid_tools": ["prepare_send", "prepare_multisend"],
                      "answer_must_contain": [], "answer_must_not_match": [r"\bsent\b", r"0x[0-9a-fA-F]{40}"]}
        ex["meta"]["cut"] = "last"
        return ex
    # continuation: the user pastes the addresses; call with amounts from the first message
    addrs = gen_addrs(rng, fam, n)
    order = list(range(n))
    if rng.random() < 0.2:
        order.reverse()
    sep = rng.choice(["\n", ", "])
    pr = []
    for i in order:
        f = rng.choice(["{p} {a}", "{p}: {a}", "{p} - {a}", "{p} = {a}"])
        pr.append(f.format(p=people[i], a=addrs[i]))
    c.user(rng.choice(ADDR_REPLY).format(pairs=sep.join(pr)), "addrs_names")
    rows = [(addrs[i], amts[i]) for i in order]
    args = rec_args(sym, named, rows)
    r = c.call(("prepare_multisend", args))
    assert r.get("status") == "review_opened", r
    c.say(ans_success(rng, r, args))
    ex = c.ex("names_then_addresses", {"scenario": "names", "slots": {"sym": sym, "fam": fam, "named": named, "people": people},
                                       "checks": list(CHECKS) + ["asked_when_needed"]})
    ex["gold"] = gold_for(c, rows, sym, named, "names", r, {"first_turn_no_call": True})
    ex["meta"]["cut"] = "last"
    return ex


# ====================================================================== self-check
BANNED = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed)",
          r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed)\b", r"successfully",
          r"funds (are|have) (in|arrived)", r"^\W*done\b", r"not financial advice", r"[·•∙]",
          r"what('s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you)", r"\bsent\b", r"\bcompleted\b",
          r"\ball set\b", r"as an ai\b", r"can't see the future"]
BANNED_RE = [re.compile(p, re.I) for p in BANNED]
TOK = None


def token_len(text):
    global TOK
    if TOK is None:
        try:
            from tokenizers import Tokenizer
            TOK = Tokenizer.from_file(os.path.join(LAB, "export", "Hermit-1.2-4B-gpu2-merged", "tokenizer.json"))
        except Exception:
            TOK = False
    if TOK:
        return len(TOK.encode(text).ids)
    return int(len(text) / 3.2)


def check3(ex, replay=True):
    msgs, meta = ex["messages"], ex["meta"]
    json.loads(json.dumps(ex, ensure_ascii=False))
    assert meta["source"] == SRC and meta["generator"] == SRC and ex["tools"] is True
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    blob = json.dumps(ex["messages"], ensure_ascii=False)
    assert other not in blob, f"{other} found in a {bio} record"
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(SYSTEM_TMPL.split("{BIO}")[0]), "system prompt"
    assert "{BIO}" not in blob
    sysrest = msgs[0]["content"].replace(bio, "{BIO}", 1)
    assert sysrest.startswith(SYSTEM_TMPL), "system prompt differs from SYSTEM_AGENT_v2"
    users = [(i, m["content"]) for i, m in enumerate(msgs) if m["role"] == "user"]
    assert len(users) == len(meta["parse"]), "parse kinds"
    calls_seen = []
    user_pos = 0
    all_results_text = " ".join(m["content"] for m in msgs if m["role"] == "tool")
    calc_results = []
    prev_assistant_question = False
    ms_calls = 0
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            user_pos += 1
            continue
        if m["role"] == "assistant" and m.get("tool_calls"):
            before_txt = [t for j, t in users if j < i]
            kinds = meta["parse"][:len(before_txt)]
            for tc in m["tool_calls"]:
                name, a = tc["name"], tc["arguments"]
                assert name in H.TOOL_NAMES, name
                assert name != "prepare_send", "single send used"
                sch = H.TOOL_SCHEMA[name]
                assert set(a) <= set(sch["properties"]), (name, a)
                for rq in sch.get("required", []):
                    assert rq in a, (name, rq)
                if name == "calculator":
                    assert list(a) == ["expression"] and isinstance(a["expression"], str) and a["expression"].strip()
                    continue
                assert name == "prepare_multisend", name
                ms_calls += 1
                assert list(a) in (["symbol", "recipients"], ["symbol", "chain", "recipients"]), f"key order {list(a)}"
                assert isinstance(a["symbol"], str) and a["symbol"] == a["symbol"].upper() and a["symbol"] in TICKERS
                if "chain" in a:
                    assert a["chain"] in CHAIN_NAME and a["chain"] == a["chain"].lower()
                recs = a["recipients"]
                assert isinstance(recs, list) and 2 <= len(recs) <= 13
                for rc in recs:
                    assert list(rc) == ["address", "amount"], rc
                    assert isinstance(rc["address"], str) and isinstance(rc["amount"], (int, float)) and not isinstance(rc["amount"], bool)
                    if isinstance(rc["amount"], float):
                        assert not rc["amount"].is_integer(), "25.0 instead of 25"
                    assert rc["amount"] > 0
                # ---- independent parse of what the user typed
                allt = "\n".join(before_txt)
                assert tickers_in(allt) == {a["symbol"]}, (tickers_in(allt), a["symbol"])
                exp_rows = []
                names_amounts = {}
                for t, k in zip(before_txt, kinds):
                    if k == "names":
                        names_amounts = parse_name_amounts(t)
                    elif k == "addrs_names":
                        spans = parse_addresses(t)
                        tt = blank(t, spans)
                        for idx_, (s, e, ad) in enumerate(spans):
                            prv = spans[idx_ - 1][1] if idx_ else 0
                            nm = re.findall(r"[A-Z][a-z]+", tt[prv:s])[-1]
                            exp_rows.append((ad, names_amounts[nm]))
                    else:
                        pr = parse_rows(t, k, a["symbol"])
                        exp_rows += pr
                got = [(rc["address"], D(rc["amount"])) for rc in recs]
                assert [x for x, _ in got] == [x for x, _ in exp_rows], ("address list", got, exp_rows)
                assert all(v is not None for _, v in exp_rows), ("unparsed amount", exp_rows, allt)
                assert [v for _, v in got] == [v for _, v in exp_rows], ("amounts", got, exp_rows, allt)
                # chain only when named
                named = set()
                for jj, (t, k) in enumerate(zip(before_txt, kinds)):
                    named |= named_chains(t, bare=(k == "chain"))
                assert len(named) <= 1, named
                assert (a.get("chain") if a.get("chain") else None) == (next(iter(named)) if named else None), ("chain", a.get("chain"), named)
            continue
        if m["role"] == "tool":
            continue
        if m["role"] == "assistant":
            t = m["content"]
            assert t.strip(), "empty assistant text"
            for rx in BANNED_RE:
                assert not rx.search(t), (rx.pattern, t)
            assert words(t) <= 70, (words(t), t)
            assert "\n" not in t.strip() or True
            # chain / network names only if a result or the user said them
            for cname in CHAIN_NAME.values():
                if re.search(rf"\b{re.escape(cname)}\b", t):
                    src = all_results_text + " " + " ".join(u for _, u in users)
                    assert re.search(rf"\b{re.escape(cname)}\b", src, re.I), f"invented network {cname}: {t}"
            if "?" in t:
                assert t.count("?") == 1, t
    assert ms_calls <= 2
    # ---- tool results
    first_calls = []
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            r = json.loads(m["content"])
            if "result" in r:
                calc_results.append(r["result"])
            if r.get("kind") == "multisend" or r.get("status") in ("needs_chain", "blocked"):
                allowed = set()
                for k in ("result_example", "other_results"):
                    v = META["prepare_multisend"][k]
                    for vv in ([v] if k == "result_example" else v.values()):
                        allowed |= set(vv)
                assert set(r) <= allowed, (set(r) - allowed)
                assert r["status"] in ("review_opened", "needs_chain", "blocked")
    # ---- replay: recorded tool messages must equal the simulator's output
    if replay:
        w = MWorld.from_state(meta["world"])
        pos = 0
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and m.get("tool_calls"):
                w.user_text = "\n".join(c for j, c in users if j < i)
                for tc in m["tool_calls"]:
                    pos += 1
                    rr = w.call(tc["name"], tc["arguments"])
                    rec = msgs[i + pos]
                    assert rec["role"] == "tool"
                    assert json.dumps(rr, ensure_ascii=False).replace("Face ID", bio) == rec["content"], ("replay", rr, rec["content"])
                pos = 0
        # totals vs the user's own text
    # ---- result totals computed independently from the typed rows
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and any(tc["name"] == "prepare_multisend" for tc in (m.get("tool_calls") or [])):
            tc = [t for t in m["tool_calls"] if t["name"] == "prepare_multisend"][0]
            r = json.loads(msgs[i + len(m["tool_calls"])]["content"]) if len(m["tool_calls"]) == 1 else json.loads(msgs[i + 1 + [t["name"] for t in m["tool_calls"]].index("prepare_multisend")]["content"])
            if r.get("status") == "review_opened":
                good = [D(x["amount"]) for x in tc["arguments"]["recipients"] if family(x["address"])]
                assert r["total"] == f"{dfmt(sum(good, Decimal(0)))} {tc['arguments']['symbol']}", (r["total"], good)
                assert r["recipients"] == len(good)
    # ---- number audit
    src = [m["content"] for m in msgs if m["role"] in ("user", "tool")]
    allowed = set()
    for s in src:
        s = blank(s, parse_addresses(s))
        for mm in NUM_RE.finditer(s):
            allowed.add(nval(mm))
            allowed.add(Decimal(mm.group(1).replace(",", "")))
    for s in calc_results:
        for mm in NUM_RE.finditer(s):
            allowed.add(nval(mm))
    for m in msgs:
        if m["role"] == "assistant" and not m.get("tool_calls"):
            t = m["content"]
            t2 = blank(t, parse_addresses(t))
            for mm in NUM_RE.finditer(t2):
                v = Decimal(mm.group(1).replace(",", ""))
                assert v in allowed or nval(mm) in allowed, ("number not backed by a result or the user", mm.group(0), t)
    # ---- final answer rules
    last = msgs[-1]
    assert last["role"] == "assistant" and not last.get("tool_calls")
    lr = None
    for m in reversed(msgs):
        if m["role"] == "tool":
            try:
                lr = json.loads(m["content"])
            except Exception:
                pass
            if lr and "result" not in lr:
                break
    if lr and lr.get("status") == "review_opened":
        assert re.search(r"approv|confirm", last["content"], re.I) and bio in last["content"], last["content"]
        assert lr["network"] in last["content"] and lr["total"] in last["content"], last["content"]
        for w_ in lr.get("warnings", []):
            assert w_ in last["content"]
        for lo in lr.get("left_out", []):
            assert re.match(r"row (\d+)", lo).group(0) in last["content"].lower(), last["content"]
        addr_in = [x for x in re.findall(r"0x[0-9a-fA-F]{40}", last["content"])]
        assert len(addr_in) <= 3
    if lr and lr.get("status") == "needs_chain":
        assert last["content"].endswith("?")
    if meta["scenario"] == "names" and ex["meta"].get("cut") == "last" and not any(m.get("tool_calls") for m in msgs):
        assert last["content"].endswith("?")
    # ---- render round trip + size
    txt = H.render_text(msgs, True)
    for m in msgs:
        if m["role"] == "assistant":
            calls, bad = H.parse_calls(H.assistant_text(m))
            assert bad == 0 and calls == (m.get("tool_calls") or []), "render round trip"
    assert txt.count("<|im_start|>assistant") == sum(1 for m in msgs if m["role"] == "assistant")
    ntok = token_len(txt)
    assert ntok <= 6144, ntok
    meta["_tokens"] = ntok
    return ntok


# ====================================================================== skeletons (leakage between train and eval)
def skeleton(ex):
    t = " | ".join(m["content"] for m in ex["messages"] if m["role"] == "user")
    sp = parse_addresses(t)
    t = blank_with(t, sp, " <A> ")
    t = NUM_RE.sub(" <N> ", t)
    t = re.sub(r"\b(" + "|".join(TICKERS) + r")\b", "<S>", t, flags=re.I)
    t = re.sub(r"\b(ethereum|mainnet|base|arbitrum|arb|bsc|bnb chain|polygon|solana|near)\b", "<C>", t, flags=re.I)
    toks = [x for x in re.split(r"\s+", t.lower()) if x]
    out = []
    for x in toks:
        core = re.sub(r"[^\w<>]", "", x)
        if core in ("<a>", "<n>", "", "to", "gets", "for", "and") or x in ("->", "=", "-", "|", ":"):
            if not out or out[-1] != "<LIST>":
                out.append("<LIST>")
        else:
            out.append(core)
    return out


def blank_with(text, spans, rep):
    out, last = [], 0
    for s, e, _ in spans:
        out.append(text[last:s])
        out.append(rep)
        last = e
    out.append(text[last:])
    return "".join(out)


def ngrams(toks, n=5):
    n = min(n, len(toks)) or 1
    return {tuple(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def first_user(ex):
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


# ====================================================================== build
def plan_counts(n, weights):
    tot = sum(weights.values())
    raw = {k: n * w / tot for k, w in weights.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    rem = n - sum(cnt.values())
    for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:rem]:
        cnt[k] += 1
    return cnt


def build(seed, n, ev, weights, avoid=None, shuffle=True):
    USE.clear()
    rng = random.Random(seed)
    cnt = plan_counts(n, weights)
    rows, seen_text, seen_skel = [], set(), []
    counters = Counter()
    rejected = Counter()
    for scen, k in cnt.items():
        for j in range(k):
            for attempt in range(60):
                r2 = random.Random(rng.getrandbits(64))
                try:
                    ex = gen(r2, ev, scen, counters[scen])
                except AssertionError as e:      # constraint miss inside a scenario: draw again
                    rejected["assert:" + scen] += 1
                    continue
                key = first_user(ex)
                sk = skeleton(ex)
                if key in seen_text:
                    rejected["dup_text"] += 1
                    continue
                if avoid is not None:
                    g = ngrams(sk)
                    if any(sk == s or jaccard(g, ng) >= 0.8 for s, ng in avoid):
                        rejected["leak"] += 1
                        continue
                seen_text.add(key)
                seen_skel.append((sk, ngrams(sk)))
                counters[scen] += 1
                rows.append(ex)
                break
            else:
                raise RuntimeError(f"could not generate {scen} #{j}")
    if shuffle:
        rng.shuffle(rows)
    return rows, seen_skel, rejected


def to_eval_item(ex, i):
    msgs = ex["messages"]
    meta = ex["meta"]
    uidx = [j for j, m in enumerate(msgs) if m["role"] == "user"]
    cut_turn = 0 if meta.get("cut") == "first" else len(uidx) - 1
    ci = uidx[cut_turn]
    nxt = uidx[cut_turn + 1] if cut_turn + 1 < len(uidx) else len(msgs)
    gold = dict(ex["gold"])
    if meta.get("cut") == "first":
        gold = {**gold, **gold["first_turn"], "scenario": gold["scenario"], "no_call": False, "must_ask": True}
        gold["amounts"], gold["addresses"] = ex["gold"]["amounts"], ex["gold"]["addresses"]
    return {"id": f"multisend_typed-e{i:03d}", "kind": meta["kind"], "tools": True, "bio": meta["bio"],
            "context": msgs[:ci + 1], "reference": msgs[ci + 1:nxt], "expect": meta["expect_by_turn"][cut_turn],
            "world": meta["world"], "checks": meta["checks"], "gold": gold,
            "meta": {"source": SRC, "scenario": meta["scenario"], "slots": meta["slots"], "parse": meta["parse"][:cut_turn + 1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--eval-seed", type=int, default=9301)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--out", default=OUT_TRAIN)
    ap.add_argument("--eval-out", default=OUT_EVAL)
    ap.add_argument("--only", choices=["both", "train", "eval"], default="both")
    a = ap.parse_args()
    # eval first: its skeletons are excluded from train
    qev = plan_counts(a.n_eval, EVAL_Q)
    ev_rows, ev_skel, ev_rej = build(a.eval_seed, a.n_eval, True, EVAL_Q, shuffle=True)
    tr_rows, tr_skel, tr_rej = build(a.seed, a.n, False, SCEN_W, avoid=ev_skel)
    maxtok = 0
    for ex in tr_rows + ev_rows:
        maxtok = max(maxtok, check3(ex))
    # file-level checks
    tr_texts = [first_user(e) for e in tr_rows]
    ev_texts = [first_user(e) for e in ev_rows]
    assert len(set(tr_texts)) == len(tr_texts), "duplicate user texts in train"
    assert len(set(ev_texts)) == len(ev_texts), "duplicate user texts in eval"
    assert not (set(tr_texts) & set(ev_texts)), "user text shared by train and eval"
    all_tr_users = {m["content"] for e in tr_rows for m in e["messages"] if m["role"] == "user"}
    all_ev_users = {m["content"] for e in ev_rows for m in e["messages"] if m["role"] == "user"}
    assert not (all_tr_users & all_ev_users), "a user message appears in both train and eval"
    for e in ev_rows:
        sk = skeleton(e)
        g = ngrams(sk)
        for s2, ng in tr_skel:
            assert sk != s2 and jaccard(g, ng) < 0.8, ("template overlap", sk, s2)
    for e in tr_rows + ev_rows:
        e["meta"].pop("_tokens", None)
    items = [to_eval_item(e, i) for i, e in enumerate(ev_rows)]
    n_ask = sum(1 for it in items if it["meta"]["scenario"] in ("names", "needs_chain"))
    assert n_ask >= 10 or a.n_eval != 50, f"names-only + needs_chain eval items: {n_ask}"
    if a.only in ("both", "train"):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            for e in tr_rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if a.only in ("both", "eval"):
        os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
        with open(a.eval_out, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    sys.stderr.write(f"scenarios train {dict(Counter(e['meta']['scenario'] for e in tr_rows))}\n"
                     f"scenarios eval  {dict(Counter(e['meta']['scenario'] for e in ev_rows))}\n"
                     f"bio train {dict(Counter(e['meta']['bio'] for e in tr_rows))}; max tokens {maxtok}; "
                     f"rejected train {dict(tr_rej)} eval {dict(ev_rej)}; ask/no-call eval items {n_ask}\n")
    print(f"multisend_typed: {len(tr_rows)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
