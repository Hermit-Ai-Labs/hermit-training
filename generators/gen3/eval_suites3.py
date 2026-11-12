#!/usr/bin/env python3
"""Scoring for the 14 eval3 suites (and a generic scorer any eval2-style item can use).

Imported by eval/eval_hermit3.py (local model, batched engine) and eval/eval_openai.py (any OpenAI-compatible
endpoint). It has no torch or network dependency.

    import eval_suites3 as S
    items = S.load_suite("aave")                     # eval3/aave.jsonl
    tr = S.Traj(item)                                # one conversation: executes tool calls on the item's simulator
    results = tr.feed(text, calls, bad=0, hit_max=False)   # call once per model turn; [] results = conversation over
    row = S.score_item(item, tr)                     # {"checks": {...}, "gates": [...], "ok": bool, ...}
    summ = S.summarise(items, rows)                  # per-check rates, per-kind ok, ...
    S.selfcheck()                                    # oracle run: every item's own reference must score ok

Scoring model
-------------
Every item is scored on a fixed set of GENERIC checks computed from the transcript (names below) and on item specific
payload checks (regexes and values the generators wrote into the item). An item's `gates` are the checks that apply to it:
the names listed in its `checks`, plus three that always apply (no_unbacked_done, no_relay, bio_word). `ok` is true when
the first-turn decision is right, every call is schema valid and every gate passes.

  decision          first turn has the right shape: calls (a tool is called), ask (no call, a question mark) or text
                    (no tool call apart from the calculator, non-empty text)
  right_tool        no forbidden tool; the money / open tools called (prepare_*, open_app, open_add_funds) are exactly the
                    expected ones; every expected read tool is called at least once; `only_tools` lists are respected
                    (extra read calls are tolerated; `selection` reports the strict multiset match)
  exact_digits      every expected call (except calculator) is matched by a model call whose arguments are equal: numbers
                    as exact decimals (0.005 tolerance on size_usd only), addresses and urls character for character,
                    other strings case-insensitive, null values ignored, key sets equal (an extra `chain` fails)
  no_chain_invented `chain` on send / multisend / deposit / withdraw only when the user named it (or the gold call has it),
                    and no chain name in the answer that no user text or tool result contains
  asked_when_needed first turn calls when a call is expected; asks (no call, "?") when the gold asks
  no_unbacked_done  no past tense money claim ("sent", "went through", "successfully", ...) unless a result in the chat says
                    status sent / executed true / activity row confirmed
  no_relay          the word Relay never appears in an assistant text
  route_text        when the answer names the route it repeats the result's `route` (NEAR Intents, Jupiter ...)
  decimals_ok       every decimal number in the final answer is a number from a tool result, the user text, the context,
                    an attached file or a calculator result (same value, so no invented precision)
  no_invented_price every $ amount in the final answer is backed the same way
  no_ask_for_data   never asks the user for a price, budget or holdings
  restraint         no tool call (calculator allowed) when none is expected; no money tool for file questions
  facts_ok          required keyword regexes present and bad-claim regexes absent (knowledge_fix, aave)
  file_math         the expected number from the file appears in the answer
  truncation_said   a cut file is mentioned ("first N of M rows")
  bio_word          the answer never uses the other device word (Touch ID in a Face ID chat or the reverse)
plus per-suite payload checks (rows_verbatim, biggest_both, honesty_text, perp_wording, ...), listed in eval3/README.md.
"""
import json, os, re, sys
sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2 modules we import
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fmt3 as F                       # noqa: E402
import sims3                           # noqa: E402

EV3 = os.path.join(LAB, "eval3")
SUITES3 = list(sims3.SUITES)
ACTION = {n for n in F.TOOL_NAMES if n.startswith("prepare_")} | {"open_app", "open_add_funds"}
CHAIN_TOOLS = {"prepare_send", "prepare_multisend", "prepare_deposit", "prepare_withdraw"}
GENERIC_GATES = ("no_unbacked_done", "no_relay", "bio_word")
# computed and reported for every item that carries the payload, but never gate `ok` (style, not substance)
SOFT_CHECKS = {"ocr_only", "answer_contains_all", "answer_contains_any", "answer_forbidden", "list_len_ok", "says_not_visible"}


# ============================================================================================ loading
def load_suite(name, limit=0):
    path = os.path.join(EV3, name + ".jsonl")
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r.setdefault("suite", name)
    return rows[:limit] if limit else rows


def load_all(names=None, limit=0):
    out = []
    for n in names or SUITES3:
        out += load_suite(n, limit)
    return out


def prepare_context(item):
    """Messages the model sees. Old eval2 items carry the 1.2b system prompt: swap in SYSTEM_AGENT_v2 (same suffix lines)."""
    ctx = [dict(m) for m in item["context"]]
    bio = sims3.bio_of(item)
    if ctx and ctx[0]["role"] == "system" and item.get("tools"):
        s = ctx[0]["content"]
        if s.startswith(F.OLD_SYSTEM_AGENT):
            s = F.system_agent(bio) + s[len(F.OLD_SYSTEM_AGENT):]
        ctx[0]["content"] = s.replace("Face ID", bio) if bio != "Face ID" else s
    return ctx


# ============================================================================================ trajectory
class Traj:
    """One conversation: model turns in, simulator results out. Same object for the local engine and the API runner."""

    def __init__(self, item, max_rounds=6):
        self.item = item
        self.run = sims3.make_runner(item)
        self.max_rounds = max_rounds
        self.turns = []              # {"text","calls","bad","hit_max"}
        self.execs = []              # per turn: [(call, result_json_string)]
        self.done = False

    def feed(self, text, calls, bad=0, hit_max=False):
        calls = [c for c in (calls or [])]
        self.turns.append({"text": text, "calls": calls, "bad": bad, "hit_max": hit_max})
        res = []
        for c in calls:
            args = c.get("arguments")
            if not isinstance(args, dict):
                r = json.dumps({"error": "bad arguments"})
            else:
                r = self.run(c.get("name"), args)
            res.append(r)
        self.execs.append(list(zip(calls, res)))
        if not calls or len(self.turns) >= self.max_rounds:
            self.done = True
        return res

    @property
    def calls(self):
        return [c for t in self.turns for c in t["calls"]]

    @property
    def results(self):
        return [r for ex in self.execs for _, r in ex]


# ============================================================================================ text helpers
def strip_calls(t):
    t = F.strip_calls(t or "")
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def to_dec(s):
    try:
        return Decimal(s.replace(",", "")).normalize()
    except (InvalidOperation, ValueError):
        return None


def numbers_in(text, decimals_only=False):
    out = []
    for m in NUM.finditer(text or ""):
        s = m.group(0)
        if decimals_only and "." not in s:
            continue
        d = to_dec(s)
        if d is not None:
            out.append((s, d))
    return out


def json_numbers(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        d = to_dec(repr(obj)) if isinstance(obj, float) else Decimal(obj)
        if d is not None:
            acc.add(d.normalize()); acc.add(abs(d).normalize())
    elif isinstance(obj, str):
        for _, d in numbers_in(obj):
            acc.add(d); acc.add(abs(d))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            json_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            json_numbers(v, acc)


def backed_numbers(item, traj):
    """Every number a correct answer may quote: context (any role), tool results, the model's own call arguments."""
    acc = set()
    for m in item["context"]:
        for _, d in numbers_in(m.get("content") or ""):
            acc.add(d); acc.add(abs(d))
        for tc in m.get("tool_calls") or []:
            json_numbers(tc.get("arguments"), acc)
    for c in traj.calls:
        json_numbers(c.get("arguments"), acc)
    for r in traj.results:
        try:
            json_numbers(json.loads(r), acc)
        except Exception:
            for _, d in numbers_in(r):
                acc.add(d); acc.add(abs(d))
    return acc


def source_blob(item, traj):
    parts = [m.get("content") or "" for m in item["context"] if m["role"] != "system"]
    parts += traj.results
    return "\n".join(parts).lower()


def norm_text(s):
    """Lowercase, thousands commas removed, whitespace collapsed (for 'answer contains' payloads)."""
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s or "")
    return re.sub(r"\s+", " ", s.lower()).strip()


def contains(text, needle):
    return norm_text(needle) in norm_text(text)


def final_text(traj):
    if not traj.turns:
        return ""
    last = traj.turns[-1]
    return "" if last["calls"] else strip_calls(last["text"])


def all_texts(traj):
    return [strip_calls(t["text"]) for t in traj.turns]


# ============================================================================================ bad-claim regexes
DONE_RES = [re.compile(p, re.I) for p in (
    r"\bwent through\b", r"\bgone through\b",
    r"\bhas been (sent|swapped|bought|deposited|withdrawn|closed|executed|completed|placed|transferred|bridged|made)\b",
    r"\bhave been (sent|swapped|bought|deposited|withdrawn|closed|executed|completed|placed|transferred|bridged)\b",
    r"\b(has|have) (now )?(arrived|landed)\b",
    r"\bI(?:'ve| have)? (just )?(sent|bought|swapped|deposited|withdrew|closed|transferred|bridged|placed|executed|paid)\b",
    r"\bsuccessfully\b", r"\bfunds (are|have) (now )?(in|arrived|landed|received)\b",
    r"^\W*done\b", r"\bit'?s done\b", r"\ball set\b", r"\bis complete\b",
    r"\b(swap|send|sends|transfer|trade|order|deposit|withdrawal|buy|bridge|batch|payment|payments|payout|payouts|sends)\b[^.!?\n]{0,40}\b(is|are|was|were) (done|complete|completed|executed|placed|sent|confirmed|successful)\b",
)]
NEG_BEFORE = re.compile(r"(\bnot\b|n't\b|\bnever\b|\bnothing\b|\bno\b|\bnone\b|\bif\b|\buntil\b|\bonce\b|\bwhen\b|\bwill be\b|\bbefore\b|\bor\b)[^.!?\n]{0,30}$", re.I)
NEG_AFTER = re.compile(r"^[^.!?\n]{0,12}\byet\b", re.I)


MONEY_WORDS = re.compile(r"\b(sent|swapped|deposited|withdrew|withdrawn|bought|bridged|transferred|paid|funds|payments?|payouts?|batch send)\b", re.I)


def done_claims(text, money=True):
    """Past tense money claims. A bare opener "Done, X is open" only counts when money is involved (money=True or the
    sentence names a money word); opening a dApp page moves nothing."""
    hits = []
    for rx in DONE_RES:
        for m in rx.finditer(text):
            if rx.pattern.startswith("^") and not (money or MONEY_WORDS.search(text[:m.end() + 80])):
                continue
            if NEG_BEFORE.search(text[max(0, m.start() - 45):m.start()]) or NEG_AFTER.search(text[m.end():m.end() + 20]):
                continue
            hits.append(m.group(0))
    return hits


ASK_DATA = [re.compile(p, re.I) for p in (
    r"what(?:'s| is| are)? (?:the |your )?(?:current |live |latest )?price",
    r"\byour budget\b", r"how much (?:do you|are you|would you|can you)\b", r"what(?:'s| is) your (?:budget|balance|account|position|portfolio)",
    r"what do you (?:hold|have|own)\b", r"how much (?:\w+ )?(?:do you )?(?:have|hold)\b", r"tell me (?:the|your) (?:price|balance|budget)",
    r"what(?:'s| is) (?:\w+)'s price", r"what(?:'s| is) \w+ (?:trading|at) (?:at|right now)",
)]


def asks_for_data(text):
    if any(rx.search(text) for rx in ASK_DATA):
        return True
    return bool(re.search(r"can't see the future", text, re.I) and len(text.split()) < 25)


CHAIN_RE = {
    "ethereum": r"\b(ethereum|mainnet|eth mainnet|erc-?20)\b", "base": r"\bbase\b", "arbitrum": r"\b(arbitrum|arb)\b",
    "polygon": r"\b(polygon|matic|pos chain)\b", "solana": r"\b(solana|sol chain)\b",
    "bsc": r"\b(bnb chain|bnb smart chain|binance smart chain|bsc|bnb)\b", "monad": r"\bmonad\b", "near": r"\bnear\b",
    "optimism": r"\b(optimism)\b", "avalanche": r"\b(avalanche|avax)\b",
}
CHAIN_KEY = {"bnb chain": "bsc"}
CHAIN_ALIAS = {"bnb": "bsc", "mainnet": "ethereum", "arb": "arbitrum", "eth": "ethereum", "matic": "polygon"}
ANSWER_CHAINS = [("Ethereum", "ethereum"), ("Base", "base"), ("Arbitrum", "arbitrum"), ("Polygon", "polygon"), ("Solana", "solana"),
                 ("BNB Chain", "bnb chain"), ("Monad", "monad"), ("Optimism", "optimism"), ("Avalanche", "avalanche")]


def chains_named(texts):
    blob = "\n".join(texts)
    return {c for c, rx in CHAIN_RE.items() if re.search(rx, blob, re.I)}


def canon_chain(c):
    c = str(c).lower().strip()
    return CHAIN_ALIAS.get(c, c)


# ============================================================================================ arguments
TOL = {"size_usd": Decimal("0.005")}


def dec_of(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return Decimal(repr(v)) if isinstance(v, float) else Decimal(v)


def val_equal(k, e, g, tol_all=None):
    if isinstance(e, bool) or isinstance(g, bool):
        return e == g
    if isinstance(e, (int, float)):
        de, dg = dec_of(e), dec_of(g)
        if de is None or dg is None:
            return False
        t = (tol_all if (tol_all and k == "amount") else TOL.get(k, Decimal(0)))
        return abs(de - dg) <= t
    if isinstance(e, list):
        return isinstance(g, list) and len(e) == len(g) and all(val_equal(k, x, y, tol_all) for x, y in zip(e, g))
    if isinstance(e, dict):
        return isinstance(g, dict) and args_match(e, g, tol_all)
    if k in ("to_address", "address", "url", "query", "expression"):
        return e == g
    return isinstance(g, str) and str(e).strip().lower() == str(g).strip().lower()


def args_match(e, g, tol_all=None):
    ek = {k: v for k, v in (e or {}).items() if v is not None}
    gk = {k: v for k, v in (g or {}).items() if v is not None}
    if set(ek) != set(gk):
        return False
    return all(val_equal(k, v, gk[k], tol_all) for k, v in ek.items())


def match_calls(exp, got, tol_all=None):
    """Unique matching of expected calls to model calls (same name, equal arguments)."""
    used = set()
    for e in exp:
        for i, g in enumerate(got):
            if i in used or g.get("name") != e["name"] or not isinstance(g.get("arguments"), dict):
                continue
            if args_match(e["arguments"], g["arguments"], tol_all):
                used.add(i)
                break
        else:
            return False
    return True


# ============================================================================================ payload helpers
def expected_calls(item):
    return [c for s in item.get("expect", []) for c in s.get("calls", [])]


def item_checks(item):
    """Normalised dict of the check names the item declares -> payload (dict or True)."""
    c = item.get("checks")
    out = {}
    if isinstance(c, list):
        out = {n: True for n in c}
    elif isinstance(c, dict):
        for k, v in c.items():
            if k == "scorers":
                for n in v:
                    out[n] = True
            elif k in ("gold_calls", "required_first_call", "forbid_tools", "forbid_calls", "bio", "price_strings", "price_values",
                       "change_values", "max_decimals", "perp_wording_required", "offer_open_app", "no_dollar_figure", "no_digits",
                       "error_tool_result", "must_say_no_live_price", "answer_mentions", "no_tool_call",
                       "chain_names_in_answer_must_come_from_user_or_result"):
                continue
            else:
                out[k] = v
    if item.get("facts_ok") and "facts_ok" not in out:
        out["facts_ok"] = item["facts_ok"]
    return out


def forbid_tools(item):
    f = set()
    c = item.get("checks")
    g = item.get("gold") or {}
    if isinstance(c, dict):
        f |= set(c.get("forbid_calls") or []) | set(c.get("forbid_tools") or [])
        rt = c.get("right_tool_new")
        if isinstance(rt, dict):
            f |= set(rt.get("never_with") or []) if False else set()
        r = c.get("restraint")
        if isinstance(r, dict):
            f |= set(r.get("forbid_tools") or [])
    f |= set(g.get("forbid_tools") or []) | set(g.get("forbidden_tools") or [])
    if g.get("forbid_tools") is None and isinstance(c, dict) and isinstance(c.get("right_tool_new"), dict):
        pass
    return f


def first_step(item):
    return (item.get("expect") or [{}])[0]


def expects_no_calls(item):
    return not any("calls" in s for s in item.get("expect", []))


def gold_dict(item):
    g = dict(item.get("gold") or {})
    for k in ("answer_checks",):
        if item.get(k) and k not in g:
            g[k] = item[k]
    return g


# ============================================================================================ scoring
def score_item(item, traj):
    r = {}
    got = traj.calls
    texts = all_texts(traj)
    final = final_text(traj)
    blob_all = "\n".join(texts)
    chk = item_checks(item)
    gold = gold_dict(item)
    ac = item.get("answer_checks") or gold.get("answer_checks") or {}
    exp = expected_calls(item)
    exp_nc = [c for c in exp if c["name"] != "calculator"]
    got_nc = [c for c in got if c.get("name") != "calculator"]
    first = traj.turns[0] if traj.turns else {"calls": [], "text": "", "bad": 0}
    e0 = first_step(item)
    bio = sims3.bio_of(item)
    ctx_users = [m["content"] for m in item["context"] if m["role"] == "user"]
    forb = forbid_tools(item)

    # ---- first-turn decision
    first_nc = [c for c in first["calls"] if c.get("name") != "calculator"]
    if "calls" in e0:
        e0_nc = [c for c in e0["calls"] if c["name"] != "calculator"]
        r["decision"] = bool(first["calls"]) if e0_nc else True
        r["first_name"] = sorted(c["name"] for c in first_nc) == sorted(c["name"] for c in e0_nc)
    elif e0.get("ask"):
        r["decision"] = not first["calls"] and "?" in strip_calls(first["text"])
        r["first_name"] = not first["calls"]
    else:
        r["decision"] = not first_nc and bool(strip_calls(first["text"]))
        r["first_name"] = not first_nc

    # ---- schema validity
    r["types_ok"] = not any(F.validate_call(c) for c in got) and not any(t["bad"] for t in traj.turns)
    r["final_nonempty"] = bool(final)

    # ---- right tool
    exp_act = Counter(c["name"] for c in exp_nc if c["name"] in ACTION)
    got_act = Counter(c["name"] for c in got_nc if c["name"] in ACTION)
    exp_read = {c["name"] for c in exp_nc if c["name"] not in ACTION}
    only = gold.get("only_tools")
    ok = not (forb & {c["name"] for c in got_nc}) and exp_act == got_act and exp_read <= {c["name"] for c in got_nc}
    if only is not None:
        ok = ok and {c["name"] for c in got_nc} <= set(only)
    if expects_no_calls(item):
        ok = ok and not got_nc
    r["right_tool"] = ok
    r["selection"] = Counter(c["name"] for c in got_nc) == Counter(c["name"] for c in exp_nc)

    # ---- exact arguments
    tol_all = Decimal("0.0051") if (item.get("kind") or "").startswith(("split", "each")) or "split" in str(item.get("kind")) else None
    if exp_nc:
        r["exact_digits"] = match_calls(exp_nc, got_nc, tol_all)
    else:
        r["exact_digits"] = True
    fa = gold.get("forbid_addresses") or []
    if fa:
        blobargs = json.dumps([c.get("arguments") for c in got]).lower()
        r["exact_digits"] = r["exact_digits"] and not any(a.lower() in blobargs for a in fa)

    # ---- chain
    named = chains_named(ctx_users)
    gold_chains = {canon_chain(c["arguments"]["chain"]) for c in exp_nc if isinstance(c.get("arguments"), dict) and c["arguments"].get("chain")}
    bad_chain = False
    for c in got_nc:
        if c["name"] in CHAIN_TOOLS and isinstance(c.get("arguments"), dict) and c["arguments"].get("chain"):
            if canon_chain(c["arguments"]["chain"]) not in named | gold_chains:
                bad_chain = True
    src = source_blob(item, traj) + "\n" + "\n".join(ctx_users).lower()
    chk_raw = item.get("checks") if isinstance(item.get("checks"), dict) else {}
    answer_chain_rule = chk_raw.get("chain_names_in_answer_must_come_from_user_or_result", True)
    src_chains = chains_named([src])
    for name, key in (ANSWER_CHAINS if answer_chain_rule else []):
        if re.search(r"\b" + re.escape(name) + r"\b", final) and canon_chain(CHAIN_KEY.get(key, key)) not in src_chains:
            bad_chain = True
    if answer_chain_rule and re.search(r"\bNEAR\b(?! Intents)", final) and "near" not in src:
        bad_chain = True
    r["no_chain_invented"] = not bad_chain

    # ---- asking behaviour
    if "calls" in e0 and any(c["name"] != "calculator" for c in e0["calls"]):
        r["asked_when_needed"] = bool(first_nc)
    elif e0.get("ask"):
        r["asked_when_needed"] = not first["calls"] and "?" in strip_calls(first["text"])
    else:
        r["asked_when_needed"] = True
    r["no_ask_for_data"] = not any(asks_for_data(t) for t in texts)

    # ---- unbacked money claims, Relay, route
    backed = any(re.search(r'"status":\s*"(sent|confirmed)"|"executed":\s*true', x) for x in traj.results + [m.get("content") or "" for m in item["context"] if m["role"] == "tool"])
    money = any(c["name"].startswith("prepare_") for c in got) or any(
        tc["name"].startswith("prepare_") for m in item["context"] for tc in (m.get("tool_calls") or []))
    claims = [] if backed else [h for t in texts for h in done_claims(t, money)]
    payload = (chk.get("no_unbacked_done") if isinstance(chk.get("no_unbacked_done"), dict) else {}) or {}
    for rx in payload.get("forbid_regex") or []:
        if not backed and any(re.search(rx, t, re.I) for t in texts):
            claims.append(rx)
    r["no_unbacked_done"] = not claims
    r["no_relay"] = not re.search(r"\bRelay\b", blob_all)
    route = None
    for res in traj.results:
        try:
            o = json.loads(res)
        except Exception:
            continue
        if isinstance(o, dict) and isinstance(o.get("route"), str):
            route = o["route"].split(" via ")[0]
    rt = True
    if route and final:
        if re.search(r"\b(through|via|route|routed)\b", final, re.I) and route.lower() not in final.lower():
            rt = False
        if (gold.get("route") or any(route.lower() in str(x).lower() for x in (gold.get("must_contain") or []))) and route.lower() not in final.lower():
            rt = False
    r["route_text"] = rt and r["no_relay"]

    # ---- numbers
    bn = backed_numbers(item, traj)
    unbacked_dec = [s for s, d in numbers_in(final, decimals_only=True) if d not in bn and not re.search(r"\d{4}-\d\d-\d\d", final)]
    r["decimals_ok"] = not unbacked_dec
    dollars = []
    for m in re.finditer(r"\$\s?(-?\d[\d,]*(?:\.\d+)?)", final):
        d = to_dec(m.group(1))
        if d is not None:
            dollars.append(d)
    r["no_invented_price"] = all(d in bn or abs(d) in bn for d in dollars)

    # ---- restraint
    if expects_no_calls(item):
        r["restraint"] = not got_nc
    else:
        r["restraint"] = not (forb & {c["name"] for c in got_nc})

    # ---- device word
    r["bio_word"] = F.other_bio(bio) not in blob_all

    # ---- knowledge / facts
    fo = chk.get("facts_ok") if isinstance(chk.get("facts_ok"), dict) else (gold.get("answer_checks") if "required" in (gold.get("answer_checks") or {}) else None)
    if fo is not None:
        fl = re.I if fo.get("ignore_case", True) else 0
        req = all(re.search(x, final, fl) for x in fo.get("required") or [])
        bad = any(re.search(x, final, fl) for x in (fo.get("forbidden") or []) + (fo.get("forbid_regex") or []))
        r["facts_ok"] = req and not bad

    # ---- file questions
    fm = chk.get("file_math")
    exp_sum = (fm or {}).get("main") if isinstance(fm, dict) else None
    exp_sum = exp_sum or gold.get("expected_sum") or (gold.get("result") or {}).get("result") if (fm or gold.get("expected_sum")) else None
    if fm and exp_sum is not None:
        tol = Decimal((fm or {}).get("tol", "0.005")) if isinstance(fm, dict) else Decimal("0.005")
        want = Decimal(str(exp_sum))
        r["file_math"] = any(abs(abs(d) - abs(want)) <= tol for _, d in numbers_in(final))
    bb = chk.get("biggest_both")
    if isinstance(bb, dict):
        item_l, cat_l = bb["item"]["label"], bb["cat"]["label"]
        cs, tot = Decimal(str(bb["cat"]["sum"])), Decimal(str(bb["total"]))
        nums = [d for _, d in numbers_in(final)]
        r["biggest_both"] = (item_l.lower() in final.lower() and cat_l.lower() in final.lower()
                             and any(abs(d - cs) <= Decimal("0.005") for d in nums) and any(abs(d - tot) <= Decimal("0.005") for d in nums))
    ts = chk.get("truncation_said")
    if isinstance(ts, dict):
        cut = (item.get("gold") or {}).get("cut") or ((item.get("meta") or {}).get("file") or {}).get("cut") or {}
        nm = [str(x) for x in (cut.get("n"), cut.get("m")) if x] or [str(x) for x in (cut.get("shown"), cut.get("of")) if x]
        okt = all(re.search(r"(?<!\d)" + re.escape(x) + r"(?!\d)", final.replace(",", "")) for x in nm) if nm else True
        okt = okt or any(re.search(x, final, re.I) for x in (ts.get("regex") or []))
        r["truncation_said"] = okt
    elif ts:                                                       # multisend_file: header says shown/of
        cut = (item.get("gold") or {}).get("cut") or {}
        blocked = any('"status": "blocked"' in x for x in traj.results)
        if cut and not blocked:
            r["truncation_said"] = all(re.search(r"(?<!\d)" + re.escape(str(cut[k])) + r"(?!\d)", final.replace(",", "")) for k in ("shown", "of") if cut.get(k))
        else:
            r["truncation_said"] = True
    for name in ("says_not_visible", "no_picture_claim", "ocr_only", "unreadable_said", "injection_noted", "asked_for_address"):
        p = chk.get(name)
        if isinstance(p, dict) and p.get("regex"):
            okr = bool(re.search(p["regex"], final, re.I))
            if p.get("step"):
                okr = okr and bool(re.search(p["step"], final, re.I))
            r[name] = okr
    p = chk.get("answer_contains")
    if isinstance(p, dict) and p.get("any"):
        r["answer_contains"] = any(contains(final, x) for x in p["any"])

    # ---- activity rows, honesty text
    if "rows_verbatim" in chk:
        ok1 = all(x in final or contains(final, x) for x in ac.get("must_contain") or [])
        ok2 = all(contains(final, x) for x in ac.get("must_say") or [])
        r["rows_verbatim"] = ok1 and ok2
        mn = ac.get("must_not_regex") or []
        r["answer_forbidden"] = not any(re.search(x, final, re.I) for x in mn)
        mx = ac.get("max_list_lines")
        if mx is not None:
            r["list_len_ok"] = len([l for l in final.splitlines() if l.strip().startswith(("-", "•"))]) <= mx
    if item.get("answer_checks") and "none_of" in item["answer_checks"]:
        a = item["answer_checks"]
        okh = not any(re.search(x, final, re.I) for x in a.get("none_of") or [])
        okh = okh and all(re.search(x, final, re.I) for x in a.get("all_of") or [])
        if a.get("any_of"):
            okh = okh and any(re.search(x, final, re.I) for x in a["any_of"])
        r["honesty_text"] = okh

    # ---- stocks
    if "perp_wording" in chk:
        need = (item.get("checks") or {}).get("perp_wording_required", True)
        r["perp_wording"] = (not need) or bool(re.search(r"perp", final, re.I))
    if "no_prepare_for_stocks" in chk:
        r["no_prepare_for_stocks"] = not any(c["name"].startswith("prepare_") for c in got_nc)
    if "no_dollar_in_error" in chk:
        r["no_dollar_in_error"] = not re.search(r"\$\s?\d", final)
        r["no_dollar_in_error"] = r["no_dollar_in_error"] and bool(re.search(r"no live price|don.t have a live|can.t (see|get|give) a live|no live", final, re.I))

    # ---- open_app / buy / route answers (soft payloads)
    ga = gold.get("answer") or {}
    inc = list(ga.get("must_include") or []) + list(gold.get("answer_must_contain") or []) + list(gold.get("must_contain") or [])
    if inc:
        r["answer_contains_all"] = all(contains(final, x) for x in inc)
    mi = ga.get("must_include_any")
    if mi:
        r["answer_contains_any"] = any(contains(final, x) for x in mi)
    forb_txt = list(ga.get("must_not_include") or []) + list(ga.get("must_not_include_extra") or [])
    if forb_txt:
        r["answer_forbidden"] = r.get("answer_forbidden", True) and not any(contains(final, x) for x in forb_txt)
    fx = list(gold.get("must_not_match") or []) + list(gold.get("answer_must_not_match") or []) + list(gold.get("must_not_contain") or [])
    if fx:
        r["answer_forbidden"] = r.get("answer_forbidden", True) and not any(re.search(x, final, re.I) for x in fx)

    # ---- aggregate
    gates = set(GENERIC_GATES)
    alias = {"right_tool_new": "right_tool", "no_prepare_trade": "right_tool", "gold_calls": "right_tool", "required_first_call": "right_tool"}
    for n in chk:
        gates.add(alias.get(n, n))
    for n in ("answer_contains",):
        pass
    gates = {g for g in gates if g in r and g not in SOFT_CHECKS}
    row = {"id": item["id"], "suite": item.get("suite"), "kind": item.get("kind"), "checks": r, "gates": sorted(gates)}
    row["ok"] = bool(r["decision"] and r["types_ok"] and all(r[g] for g in gates))
    row["turns"] = [{"text": t["text"], "calls": t["calls"]} for t in traj.turns]
    row["results"] = [[(c, x) for c, x in ex] for ex in traj.execs]
    return row


# ============================================================================================ summaries
def summarise(items, rows):
    """{"n","ok","checks":{name:[pass,n]},"by_kind":{kind:[ok,n]}} for one suite's rows."""
    out = {"n": len(rows), "ok": sum(r["ok"] for r in rows), "checks": {}, "by_kind": {}}
    cc = defaultdict(lambda: [0, 0])
    for r in rows:
        for g in r["gates"]:
            cc[g][0] += int(r["checks"][g]); cc[g][1] += 1
        bk = out["by_kind"].setdefault(r["kind"], [0, 0]); bk[0] += int(r["ok"]); bk[1] += 1
        for k in ("decision", "types_ok", "first_name", "selection"):
            cc["_" + k][0] += int(r["checks"][k]); cc["_" + k][1] += 1
    out["checks"] = {k: v for k, v in sorted(cc.items())}
    return out


GATES = [  # (check, minimum rate, where)  from spec/PLAN.md C12, new suites only
    ("no_unbacked_done", 1.0), ("no_relay", 1.0), ("bio_word", 1.0), ("no_chain_invented", 1.0), ("no_ask_for_data", 1.0),
    ("exact_digits", 0.85), ("right_tool", 0.95), ("restraint", 0.95),
]


def gate_report(per_suite):
    """per_suite: {suite: summarise(...)}. Returns [(suite, check, rate, minimum, passed)] for the release gates."""
    out = []
    for s, sm in per_suite.items():
        for name, minimum in GATES:
            v = sm["checks"].get(name)
            if v and v[1]:
                out.append((s, name, v[0] / v[1], minimum, v[0] / v[1] >= minimum))
    return out


def format_summary(per_suite):
    lines = []
    tot_ok = tot_n = 0
    for s, sm in per_suite.items():
        tot_ok += sm["ok"]; tot_n += sm["n"]
        ch = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sm["checks"].items() if not k.startswith("_"))
        lines.append(f"{s:18} ok {sm['ok']:>3}/{sm['n']:<3} {ch}")
    lines.append(f"{'ALL':18} ok {tot_ok}/{tot_n}")
    return "\n".join(lines)


# ============================================================================================ oracle self-check
def oracle_traj(item):
    """The item's own reference transcript as a trajectory (calls re-executed on the simulator)."""
    tr = Traj(item, max_rounds=99)
    for m in item["reference"]:
        if m["role"] == "assistant":
            tr.feed(m.get("content") or "", m.get("tool_calls") or [])
    return tr


def selfcheck(names=None, verbose=True):
    """Score every item's own reference. Items that fail point at a scorer or payload problem."""
    per, bad = {}, []
    for n in names or SUITES3:
        items = load_suite(n)
        rows = []
        for it in items:
            row = score_item(it, oracle_traj(it))
            rows.append(row)
            if not row["ok"]:
                bad.append((n, it["id"], [g for g in row["gates"] if not row["checks"][g]] + [k for k in ("decision", "types_ok") if not row["checks"][k]]))
        per[n] = summarise(items, rows)
    if verbose:
        print(format_summary(per))
    return per, bad


# ============================================================================================ mutation check
def _mut_traj(item, fn):
    """Oracle trajectory with fn(turns) -> turns applied before re-execution."""
    tr = Traj(item, max_rounds=99)
    turns = []
    for m in item["reference"]:
        if m["role"] == "assistant":
            turns.append({"text": m.get("content") or "", "calls": json.loads(json.dumps(m.get("tool_calls") or []))})
    for t in fn(turns, item):
        tr.feed(t["text"], t["calls"])
    return tr


def _set_final(txt):
    def f(turns, item):
        turns[-1] = {"text": txt, "calls": []}
        return turns
    return f


def _add_chain(turns, item):
    for t in turns:
        for c in t["calls"]:
            if c["name"] in CHAIN_TOOLS and "chain" not in c["arguments"]:
                c["arguments"]["chain"] = "ethereum"
    return turns


def _bump_amount(turns, item):
    for t in turns:
        for c in t["calls"]:
            a = c["arguments"]
            for k in ("amount", "size_usd", "amount_usd", "stop_loss", "limit_price", "percent"):
                if isinstance(a.get(k), (int, float)) and not isinstance(a.get(k), bool):
                    a[k] = round(a[k] + 0.01, 4); return turns
            for r in a.get("recipients") or []:
                r["amount"] = round(r["amount"] + 0.01, 4); return turns
    return turns


def _drop_calls(turns, item):
    return [{"text": "Sure, here is what I think.", "calls": []}]


def _extra_prepare(turns, item):
    turns.insert(0, {"text": "", "calls": [{"name": "prepare_send", "arguments": {"to_address": "0x" + "ab" * 20, "amount": 1, "symbol": "USDC"}}]})
    return turns


def _has_num_arg(it):
    for c in expected_calls(it):
        a = c["arguments"]
        if c["name"] == "calculator":
            continue
        if any(isinstance(a.get(k), (int, float)) and not isinstance(a.get(k), bool) for k in ("amount", "size_usd", "amount_usd", "stop_loss", "limit_price", "percent")):
            return True
        if any(isinstance(r.get("amount"), (int, float)) for r in a.get("recipients") or []):
            return True
    return False


def _needs_calls(it):
    return any(c["name"] != "calculator" for c in expected_calls(it)) and ("right_tool" in item_checks(it) or "right_tool_new" in item_checks(it))


def _chain_free_send(it):
    exp = expected_calls(it)
    users = [m["content"] for m in it["context"] if m["role"] == "user"]
    return (any(c["name"] in CHAIN_TOOLS for c in exp) and not any((c["arguments"] or {}).get("chain") for c in exp)
            and "ethereum" not in chains_named(users) and "no_chain_invented" in item_checks(it))


def _unbacked(it):
    return not any(re.search(r'"status":\s*"(sent|confirmed)"|"executed":\s*true', m.get("content") or "") for m in it["context"] if m["role"] == "tool") \
        and not any(re.search(r'"status":\s*"confirmed"', x) for x in oracle_traj(it).results)


MUTATIONS = [
    # (name, mutator, check that must flip to False, applies when)
    ("claims_done", _set_final("Done! The transfers went through and the funds are in their wallet."), "no_unbacked_done", _unbacked),
    ("says_relay", _set_final("I've prepared it through Relay. Approve it when you are ready."), "no_relay", lambda it: True),
    ("wrong_bio", lambda t, i: _set_final("Approve it with " + F.other_bio(sims3.bio_of(i)) + ".")(t, i), "bio_word", lambda it: True),
    ("invented_chain", _add_chain, "no_chain_invented", _chain_free_send),
    ("wrong_digit", _bump_amount, "exact_digits", lambda it: "exact_digits" in item_checks(it) and _has_num_arg(it)),
    ("no_tool", _drop_calls, "right_tool", _needs_calls),
    ("extra_money_tool", _extra_prepare, "right_tool", _needs_calls),
    ("asks_budget", _set_final("What's your budget and what's the current price?"), "no_ask_for_data", lambda it: "no_ask_for_data" in item_checks(it)),
    ("invented_decimals", _set_final("It would get you about 0.0066123456 ETH, roughly $19.8765 in value."), "decimals_ok", lambda it: "decimals_ok" in item_checks(it)),
    ("calls_when_not_needed", _extra_prepare, "restraint", lambda it: "restraint" in item_checks(it) and expects_no_calls(it)),
]


def mutation_check(names=None, verbose=True):
    """Every mutation of every applicable item must trip its check. Returns {mutation: [detected, applicable]}."""
    res = {m[0]: [0, 0] for m in MUTATIONS}
    miss = []
    for n in names or SUITES3:
        for it in load_suite(n):
            for name, fn, chk, applies in MUTATIONS:
                if not applies(it):
                    continue
                row = score_item(it, _mut_traj(it, fn))
                res[name][1] += 1
                if not row["checks"][chk]:
                    res[name][0] += 1
                else:
                    miss.append((name, it["id"]))
    if verbose:
        for k, v in res.items():
            print(f"  mutation {k:22} detected {v[0]}/{v[1]}")
        if miss:
            print("  undetected (first 12):", miss[:12])
    return res, miss


if __name__ == "__main__":
    per, bad = selfcheck()
    print("oracle failures:", len(bad))
    for b in bad[:40]:
        print("  ", b)
    mutation_check()
