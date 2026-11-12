#!/usr/bin/env python3
"""Assemble the sft4 training and validation sets: data3/sft4_train.jsonl, data3/sft4_val.jsonl (+ data3/sft4_report.json).

Inputs (read only): data2/sft3_train.jsonl, data2/sft3_val.jsonl (the 1.2b set) and the 14 new pools data3/<name>.jsonl.
Output rows keep the sft3 shape: {"messages": [...], "tools": bool, "meta": {...}}. `tools: true` means "render with the
19-tool block"; the device word is `meta.bio` (and the system message). Render with gen3/fmt3.py (render_text(messages, tools)):
it builds the tool block for that word, exactly like the app. `meta` is slimmed to {source, kind, bio, tools_version, ...}.

What it does
  1. Legacy rows (sft3). Tools-present rows are re-rendered: system = SYSTEM_AGENT_v2 (name / hermit.md lines kept), device
     word 60% Face ID / 40% Touch ID applied to every message. Every "Relay" assumption is checked (see LEGACY below):
       * kinds the new pools replace, or whose gen2 tool shapes teach wrong things, are dropped on purpose
         (single sends with an always-present chain, swaps and quotes with 24-decimal amount_out and "via Relay", search_markets
         funding_8h_pct, open_add_funds "opened", memecoin prices, "max leverage" errors, rows with maintenance_margin);
       * read and trade results whose app shape differs (get_balances, hyperliquid_account, get_price source, get_yields chain,
         prepare_trade result and margin error) are converted to the shapes in spec/hermit_tools_v2.meta.json;
       * a number audit then drops any converted row whose answer quotes a number that no result, user text or calculator gave.
     Chat-only rows (tools false) keep SYSTEM_CHAT and are untouched except the lint.
  2. New pools: checked, slimmed, lint and leakage filtered, nothing else changed.
  3. Lint (bad claims): Relay as a route, "never write it down", "does not store" near seed/key, impermanent loss near Aave,
     .eth that "can't resolve", funding_8h, wrong device word, unsubstituted {BIO}. Hits are dropped and counted per source.
  4. Leakage: normalised text of the user messages of every eval2 and eval3 item (contexts, math questions, chat prompts,
     multi-turn scripts, notool, tools_fixed). A row is dropped if its first user message (4+ words) or a later one (12+ words)
     equals an eval user message, or its whole user sequence equals an eval item's (catches short single-turn prompts), or it
     contains an eval address. New pools also drop 5-gram Jaccard >= 0.8 near duplicates; legacy rows are exact-text only (the
     near duplicates that stay, e.g. yields_calc, are counted in the report). Generic short follow-ups ("ok do it", "yes") are
     shared by design. The build fails if a new-pool kind keeps < 90% of its rows (a legacy kind < 50%).
  5. Schema: every tool call is validated against the 19-tool registry; render and parse round trip; token cap 6144
     (Qwen3 tokenizer, with the 19-tool block).
  6. Mix: new pools x1 (about 40% of the rows), legacy core pools x1, small restraint pools repeated as in sft3
     (voice x4, strategy x2, code x3) and topped up until the tool-restraint share is at least what sft3 had.
     Val: per source, taken before repetition, 2% (min 5) of sources with 40+ unique rows; none for smaller ones.

    python gen3/assemble4.py                 # writes data3/sft4_*.jsonl
    python gen3/assemble4.py --no-tokens     # skip the tokenizer pass
"""
import argparse, hashlib, json, math, os, random, re, sys
sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2 modules we import
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import fmt3 as F                                                # noqa: E402

D2 = os.path.join(LAB, "data2")
D3 = os.path.join(LAB, "data3")
EV2 = os.path.join(LAB, "eval2")
EV3 = os.path.join(LAB, "eval3")
NEW_POOLS = ["multisend_typed", "multisend_file", "aave", "send_nochain", "route_rounding", "honesty", "trade_view", "stocks_price",
             "file_qa", "knowledge_fix", "buy_jupiter", "close_position", "activity_tool", "open_app_tool"]
REPEAT = {"restraint_voice": 4, "restraint_strategy": 2, "restraint_code": 3}      # as in sft3; everything else x1
MAX_TOKENS = 6144
BIO_SHARE_FACE = 0.6


# ============================================================================================ helpers
def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def toks(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def user_texts(msgs):
    return [m["content"] for m in msgs if m["role"] == "user"]


def msg_hash(msgs):
    return hashlib.sha1(json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dec(s):
    try:
        return Decimal(str(s).replace(",", "")).normalize()
    except (InvalidOperation, ValueError):
        return None


def plain(x):
    """Shortest plain decimal string of a number (no exponent), like the app's amount strings."""
    d = Decimal(repr(x)).normalize()
    s = format(d, "f")
    return s


# ============================================================================================ lint
LINT = [(n, re.compile(p, re.I | re.S)) for n, p in (
    ("relay", r"\bRelay\b|relay\.link|via relay|through relay|using relay"),
    ("never_write_down", r"never\s+write\s+(it|them|this|that)\s+down|(never|don'?t|do not)\s+write\s+(it|them)\s+(down\s+)?on\s+paper"),
    ("not_stored", r"(does ?n[o']t|doesn't|do not|never) store[^.]{0,80}\b(seed|key|phrase)\b|\b(seed|key|phrase)\b[^.]{0,80}(is|are) not stored"),
    ("il_aave", r"impermanent loss[^.]{0,80}\baave\b|\baave\b[^.]{0,80}impermanent loss"),
    ("eth_cant_resolve", r"\.eth[^.]{0,80}(can'?t|cannot|unable to|don'?t) (resolve|look up)|(can'?t|cannot|unable to) (resolve|look up)[^.]{0,40}\.eth"),
    ("funding_8h", r"funding_8h"),
    ("unfilled_bio", r"\{BIO\}"),
    ("middle_dot", r"[·•]"),
)]


def lint_hits(msgs, bio):
    hits = []
    for m in msgs:
        if m["role"] not in ("assistant", "tool"):
            continue
        t = m.get("content") or ""
        for name, rx in LINT:
            if name == "middle_dot" and m["role"] == "tool":
                continue
            if rx.search(t):
                hits.append(name)
    other = F.other_bio(bio)
    if bio and any(other in (m.get("content") or "") for m in msgs):
        hits.append("wrong_bio")
    return hits


# ============================================================================================ legacy conversion
# kinds dropped on purpose. reason -> kinds
DROP_KINDS = {
    "replaced by send_nochain (old sends always carry chain; gen2 result has no network)": {"send", "send_error"},
    "replaced by route_rounding (old results: 8 to 24 decimals, provider, 'via Relay' answers)": {
        "swap", "swap_direct", "swap_all", "swap_error", "quote", "quote_ask", "quote_then_prepare"},
    "search_markets shape and funding_8h_pct unit are wrong for the app (needs a core_reads regeneration)": {
        "markets_funding", "markets_funding_annual", "markets_none"},
    "open_add_funds: old answers say 'opened', the app shows a button": {"add_funds_wallet", "add_funds_hl"},
    "price_error: old error texts do not match the app's 'No Hyperliquid market' (stocks_price and tool_errors cover it)": {"price_error", "tool_error"},
}
KIND_DROP_REASON = {k: why for why, ks in DROP_KINDS.items() for k in ks}
CORE_SOURCE = {"math_tool": "core_math", "math_steps": "core_math", "user_given_price": "core_math", "chat_no_live_price": "core_reads",
               "price": "core_reads", "price_multi": "core_reads", "price_compare": "core_reads", "value_of": "core_reads",
               "balances_chain": "core_reads", "balances_chain_empty": "core_reads", "yields": "core_reads", "yields_calc": "core_reads",
               "position_math": "core_reads", "position_math_missing": "core_reads", "hl_pnl": "core_reads", "hl_liq_distance": "core_reads",
               "tool_error_retry": "core_reads", "send_scam_warn": "restraint_misc", "quote_notool": "restraint_misc"}
CHAIN_DISPLAY = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "bsc": "BNB Chain", "polygon": "Polygon",
                 "monad": "Monad", "solana": "Solana", "hyperliquid": "Hyperliquid", "near": "NEAR"}


class Drop(Exception):
    pass


def conv_get_price(o, args):
    if "error" in o:
        if str(o["error"]).startswith("unknown symbol") or str(o["error"]).startswith("price feed"):
            raise Drop("price error text differs from the app")
        return o
    if o.get("source") == "binance":
        raise Drop("memecoin price from binance (the app has no live price for these)")
    return {"change_24h_pct": o["change_24h_pct"], "price": o["price"], "source": "Hyperliquid mark price", "symbol": o["symbol"]}


def conv_get_balances(o, args):
    if "error" in o:
        return o
    rows = [{"amount": plain(b["amount"]), "chain": "bnb" if b["chain"] == "bsc" else b["chain"], "symbol": b["symbol"], "usd": b["usd"]}
            for b in o.get("balances", []) if b["amount"] > 0]
    return {"holdings": rows}


def conv_hl_account(o, args):
    if "error" in o:
        return o
    pos = [{"coin": p["coin"], "entry": p["entry"], "leverage": p["leverage"], "liquidation": p["liq"], "side": p["side"], "size": p["size"],
            "unrealised_pnl": p["unrealized_pnl"]} for p in o.get("positions", [])]
    orders = [{"coin": x["coin"], "price": x["price"], "side": x["side"], "size": x["size"]} for x in o.get("open_orders", [])]
    return {"account_value": o["account_value"], "available_usdc": o["available_usdc"], "margin_used": o["margin_used"], "open_orders": orders,
            "positions": pos}


def conv_get_yields(o, args):
    if "error" in o:
        return o
    return {"pools": [{"apy_pct": p["apy_pct"], "asset": p["asset"], "chain": CHAIN_DISPLAY.get(p["chain"], p["chain"].title()),
                       "protocol": p["protocol"], "tvl_usd": p["tvl_usd"]} for p in o.get("pools", [])]}


def conv_prepare_trade(o, args):
    if "error" in o:
        e = str(o["error"])
        need = float(args["size_usd"]) / float(args["leverage"])
        if e == "no USDC in the Hyperliquid account":
            return {"error": f"Not enough margin: 0.00 USDC available, {need:.2f} needed", "prepared": False}
        m = re.match(r"not enough margin: needs \$([\d,.]+), available \$([\d,.]+)", e)
        if m:
            return {"error": f"Not enough margin: {float(m.group(2).replace(',', '')):.2f} USDC available, {float(m.group(1).replace(',', '')):.2f} needed",
                    "prepared": False}
        raise Drop("prepare_trade error with no app equivalent (" + e.split(" ")[0] + " ...)")
    return {"status": "review_opened", "kind": "trade", "executed": False, "note": "Waiting for the user to approve with Face ID."}


CONVERT = {"get_price": conv_get_price, "get_balances": conv_get_balances, "hyperliquid_account": conv_hl_account,
           "get_yields": conv_get_yields, "prepare_trade": conv_prepare_trade}
KEEP_AS_IS = {"calculator", "position_math"}
UNSAFE_TOOLS = {"get_quote", "prepare_swap", "prepare_send", "search_markets", "open_add_funds"}


def number_audit(msgs):
    """Every $-amount, decimal and 4+ digit number in an assistant text must occur in something that came BEFORE it: the
    system text, a user turn, an earlier tool result or call, or an earlier assistant turn (follow-ups reuse figures). A
    rounded copy of a source number counts (11.06 for 11.0595). Returns the first offending number or None."""
    src = set()

    def add(text):
        for s_ in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or ""):
            d = dec(s_)
            if d is not None:
                src.add(d); src.add(abs(d))

    SCALE = {"k": Decimal(10) ** 3, "m": Decimal(10) ** 6, "b": Decimal(10) ** 9, "t": Decimal(10) ** 12}
    for m in msgs:
        t = m.get("content") or ""
        if m["role"] == "assistant" and t:
            for mm in re.finditer(r"(\$)?(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s?([kKmMbBtT]\b)?", t):
                s_ = mm.group(2)
                if not (mm.group(1) or "." in s_ or len(s_.replace(",", "").lstrip("-")) >= 4):
                    continue
                d = dec(s_)
                if d is None:
                    continue
                k = len(s_.split(".")[1]) if "." in s_ else 0
                half = Decimal(5) * Decimal(10) ** (-k - 1)                    # half a unit in the last place shown
                sc = SCALE.get((mm.group(3) or "").lower())
                if sc:
                    d, half = d * sc, half * sc
                elif d in src or abs(d) in src:
                    continue
                if any(abs(abs(x) - abs(d)) <= half for x in src if abs(x) < Decimal("1e15")):
                    continue
                return s_
        add(t)
        for tc in m.get("tool_calls") or []:
            add(json.dumps(tc["arguments"]))
    return None


def convert_legacy(r, bio):
    """sft3 row -> (new row, None) or (None, reason)."""
    meta = r["meta"]
    src, kind = meta.get("source"), meta.get("kind")
    msgs = json.loads(json.dumps(r["messages"]))
    if src == "agent":
        if kind in KIND_DROP_REASON:
            return None, "policy: " + KIND_DROP_REASON[kind]
        out_src = CORE_SOURCE.get(kind) or ("core_trade" if str(kind).startswith("trade_") else None)
        if out_src is None:
            return None, f"policy: unknown agent kind {kind}"
    elif src == "teacher":
        out_src = "restraint_teacher"
    elif src == "code_old":
        out_src = "restraint_code"
    elif src == "strategy":
        out_src = "restraint_strategy"
    elif src == "voice":
        out_src = "restraint_voice"
    else:
        return None, f"policy: unknown source {src}"
    if r["tools"]:
        names = [tc["name"] for m in msgs for tc in (m.get("tool_calls") or [])]
        if set(names) & UNSAFE_TOOLS:
            return None, "policy: calls a tool whose gen2 shape or behaviour the app does not have (" + ",".join(sorted(set(names) & UNSAFE_TOOLS)) + ")"
        queue = []
        for m in msgs:
            if m["role"] == "assistant" and m.get("tool_calls"):
                queue = [(tc["name"], tc["arguments"]) for tc in m["tool_calls"]]
                for name, a in queue:
                    if name == "position_math" and a.get("maintenance_margin"):
                        return None, "policy: position_math with maintenance_margin (formula differs from the app)"
            elif m["role"] == "tool":
                name, a = queue.pop(0) if queue else (None, {})
                try:
                    o = json.loads(m["content"])
                except Exception:
                    return None, "policy: tool result is not JSON"
                if name in CONVERT:
                    try:
                        m["content"] = json.dumps(CONVERT[name](o, a), ensure_ascii=False)
                    except Drop as e:
                        return None, "policy: " + str(e)
                elif name not in KEEP_AS_IS:
                    return None, f"policy: no converter for {name}"
        sysm = msgs[0]["content"]
        if not sysm.startswith(F.OLD_SYSTEM_AGENT):
            return None, "policy: tools row without the 1.2b agent system prompt"
        msgs[0]["content"] = F.system_agent("Face ID") + sysm[len(F.OLD_SYSTEM_AGENT):]
        for m in msgs:
            if m.get("content"):
                m["content"] = m["content"].replace("Face ID", bio)
            for tc in m.get("tool_calls") or []:
                pass
        if kind == "hl_pnl":                                # the app's position rows have no mark price: drop that clause
            for m in msgs:
                if m["role"] == "assistant" and m.get("content"):
                    m["content"] = re.sub(r",? mark \$[\d,]+(?:\.\d+)?", "", m["content"])
        if names:                                           # knowledge answers (no tool calls) quote facts, not results
            bad = number_audit(msgs)
            if bad:
                return None, "audit: " + str(kind) + ": answer quotes a number no result gave"
    else:
        sysm = msgs[0]["content"]
        has_f = any("Face ID" in (m.get("content") or "") for m in msgs)
        has_t = any("Touch ID" in (m.get("content") or "") for m in msgs)
        bio = "Touch ID" if has_t and not has_f else "Face ID"
    new = {"messages": msgs, "tools": bool(r["tools"]),
           "meta": {"source": out_src, "kind": kind or (meta.get("topic") or meta.get("bucket") or "answer"), "bio": bio, "tools_version": "v2" if r["tools"] else None,
                    "legacy": True, "orig_source": src}}
    return new, None


# ============================================================================================ new pools
def slim_new(r, pool):
    m = r["meta"]
    bio = m.get("bio") or F.detect_bio(r["messages"])
    meta = {"source": pool, "kind": m.get("kind") or pool, "bio": bio, "tools_version": "v2"}
    return {"messages": r["messages"], "tools": bool(r["tools"]), "meta": meta}


def check_new_row(row):
    """Structural checks on a new-pool row. Returns a list of problems."""
    errs = []
    msgs, bio = row["messages"], row["meta"]["bio"]
    if msgs[0]["role"] != "system" or not msgs[0]["content"].startswith(F.system_agent(bio)):
        errs.append("system prompt is not SYSTEM_AGENT_v2 with the row's device word")
    if not row["tools"]:
        errs.append("tools flag false")
    return errs


# ============================================================================================ leakage index
class LeakIndex:
    """Eval user texts. A training row leaks when (a) its first user message (4+ words) or a later one (12+ words) equals or
    nearly equals (5-gram Jaccard >= 0.8) an eval user message, (b) it contains an eval address, or (c) its whole sequence of
    user messages equals an eval item's (this catches short single-turn prompts). Generic short follow-ups ("ok do it", "yes",
    "show my balances" as a setup turn) are shared by design and are not leakage on their own."""

    def __init__(self):
        self.seqs = set()
        self.exact = {}
        self.grams = defaultdict(set)
        self.n = 0
        self.addr = set()

    def add(self, text, origin):
        if not text or not text.strip():
            return
        k = norm(text)
        self.exact.setdefault(k, origin)
        for a in re.findall(r"0x[0-9a-fA-F]{40}", text):
            self.addr.add(a.lower())
        t = toks(text)
        if len(t) >= 8:
            gs = {hash(tuple(t[i:i + 5])) for i in range(len(t) - 4)}
            idx = self.n
            self.n += 1
            self.sets = getattr(self, "sets", [])
            self.sets.append(gs)
            for g in gs:
                self.grams[g].add(idx)

    def add_seq(self, users):
        if users:
            self.seqs.add("|".join(norm(u) for u in users))

    def hit_row(self, users, fuzzy=True):
        """Reason a row leaks, or None. fuzzy=False: exact normalised text, whole sequence and addresses only (legacy rows)."""
        if "|".join(norm(u) for u in users) in self.seqs:
            return "sequence"
        for i, u in enumerate(users):
            for a in re.findall(r"0x[0-9a-fA-F]{40}", u):
                if a.lower() in self.addr:
                    return "address"
            # first message: 4+ words. Later turns ("how much do i lose if the stop gets hit") are generic by nature and
            # only count at 12+ words; a follow-up with the same context is caught by the whole-sequence rule above.
            if len(toks(u)) >= (4 if i == 0 else 12):
                h = self.hit(u, fuzzy)
                if h:
                    return h
        return None

    def hit(self, text, fuzzy=True):
        k = norm(text)
        if k in self.exact:
            return "exact"
        t = toks(text)
        if fuzzy and len(t) >= 8 and self.n:
            gs = {hash(tuple(t[i:i + 5])) for i in range(len(t) - 4)}
            cnt = Counter()
            for g in gs:
                for idx in self.grams.get(g, ()):
                    cnt[idx] += 1
            for idx, inter in cnt.items():
                if inter / (len(gs) + len(self.sets[idx]) - inter) >= 0.8:
                    return "jaccard"
        for a in re.findall(r"0x[0-9a-fA-F]{40}", text):
            if a.lower() in self.addr:
                return "address"
        return None


def build_leak_index():
    ix = LeakIndex()
    n_items = 0
    for f in ("tools.jsonl", "quote.jsonl", "notool.jsonl", "multiturn.jsonl", "chat.jsonl", "math_v2.jsonl", "math_legacy.jsonl"):
        for r in load(os.path.join(EV2, f)):
            n_items += 1
            for m in r.get("context", []) or []:
                if m["role"] == "user":
                    ix.add(m["content"], "eval2/" + f)
            ix.add_seq([m["content"] for m in r.get("context", []) or [] if m["role"] == "user"])
            for t in r.get("turns", []) or []:
                ix.add(t, "eval2/" + f)
            ix.add_seq(r.get("turns") or [])
            for k in ("question", "prompt"):
                if r.get(k):
                    ix.add_seq([r[k]])
            for k in ("question", "prompt"):
                if r.get(k):
                    ix.add(r[k], "eval2/" + f)
    for f in sorted(os.listdir(EV3)):
        if not f.endswith(".jsonl"):
            continue
        for r in load(os.path.join(EV3, f)):
            n_items += 1
            for m in r.get("context", []) or []:
                if m["role"] == "user":
                    ix.add(m["content"], "eval3/" + f)
            ix.add_seq([m["content"] for m in r.get("context", []) or [] if m["role"] == "user"])
    return ix, n_items


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=os.path.join(D3, "sft4"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--rep", action="append", default=[], help="source=N repeat override, e.g. restraint_teacher=2")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    brng = random.Random(1234)                                      # device word draws, independent of everything else
    rep = dict(REPEAT)
    for x in a.rep:
        k, v = x.split("=")
        rep[k] = int(v)
    stats = {"policy_drop": Counter(), "lint_drop": Counter(), "leak_drop": Counter(), "schema_drop": Counter(), "token_drop": Counter(),
             "dup_drop": Counter(), "new_struct_drop": Counter(), "legacy_near_duplicate_kept": Counter()}
    kind_in, kind_leak = defaultdict(Counter), defaultdict(Counter)

    # ---- old restraint share (definition used for old and new)
    def is_restraint(row):
        if not row["tools"] or any(m["role"] == "tool" or m.get("tool_calls") for m in row["messages"]):
            return False
        m = row["meta"]
        return str(m.get("source", "")).startswith("restraint") or m.get("source") in ("knowledge_fix",) or m.get("orig_source") in ("teacher", "code_old", "strategy", "voice")

    old_train = load(os.path.join(D2, "sft3_train.jsonl"))
    old_val = load(os.path.join(D2, "sft3_val.jsonl"))

    def old_is_restraint(r):
        if not r["tools"] or any(m["role"] == "tool" or m.get("tool_calls") for m in r["messages"]):
            return False
        return r["meta"].get("source") in ("teacher", "code_old", "strategy", "voice") or r["meta"].get("kind") in ("send_scam_warn", "quote_notool")
    old_share = sum(old_is_restraint(r) for r in old_train) / len(old_train)
    print(f"sft3 train: {len(old_train)} rows, tool-restraint share {old_share:.4f} ({sum(old_is_restraint(r) for r in old_train)} rows)")

    ix, n_ev = build_leak_index()
    print(f"leak index: {n_ev} eval items, {len(ix.exact)} distinct user texts, {ix.n} long texts, {len(ix.addr)} addresses")

    pools = defaultdict(list)                  # source -> rows (unique, converted)
    val_flag = {}                              # id(row) -> True if it came from the sft3 val split
    seen = set()

    # ---- legacy
    for split, rows in (("train", old_train), ("val", old_val)):
        for r in rows:
            bio = "Face ID" if brng.random() < BIO_SHARE_FACE else "Touch ID"
            new, why = convert_legacy(r, bio)
            k0 = r["meta"].get("kind") or r["meta"].get("source")
            if new is None:
                stats["policy_drop"][why.split(" (")[0][:110]] += 1
                continue
            src, kind = new["meta"]["source"], new["meta"]["kind"]
            h = msg_hash(new["messages"])
            if h in seen:
                stats["dup_drop"][src] += 1
                continue
            seen.add(h)
            kind_in[src][kind] += 1
            hits = lint_hits(new["messages"], new["meta"]["bio"])
            if hits:
                stats["lint_drop"][(src, hits[0])] += 1
                continue
            if ix.hit_row(user_texts(new["messages"]), fuzzy=False):
                stats["leak_drop"][src] += 1
                kind_leak[src][kind] += 1
                continue
            if ix.hit_row(user_texts(new["messages"]), fuzzy=True):
                stats["legacy_near_duplicate_kept"][src + "/" + kind] += 1        # template overlap with eval, numbers differ: reported
            bad = [e for m in new["messages"] for tc in (m.get("tool_calls") or []) for e in F.validate_call(tc)] if new["tools"] else []
            if bad:
                stats["schema_drop"][src] += 1
                continue
            pools[src].append(new)
            if split == "val":
                val_flag[id(new)] = True

    # ---- new pools
    for name in NEW_POOLS:
        rows = load(os.path.join(D3, name + ".jsonl"))
        if not rows:
            raise SystemExit(f"missing or empty pool {name}")
        for r in rows:
            new = slim_new(r, name)
            kind = new["meta"]["kind"]
            kind_in[name][kind] += 1
            errs = check_new_row(new)
            if errs:
                stats["new_struct_drop"][(name, errs[0])] += 1
                continue
            hits = lint_hits(new["messages"], new["meta"]["bio"])
            if hits:
                stats["lint_drop"][(name, hits[0])] += 1
                continue
            bad = [e for m in new["messages"] for tc in (m.get("tool_calls") or []) for e in F.validate_call(tc)]
            if bad:
                stats["schema_drop"][name] += 1
                continue
            h = msg_hash(new["messages"])
            if h in seen:
                stats["dup_drop"][name] += 1
                continue
            seen.add(h)
            if ix.hit_row(user_texts(new["messages"])):
                stats["leak_drop"][name] += 1
                kind_leak[name][kind] += 1
                continue
            pools[name].append(new)

    # ---- retention rule: no new-pool kind may lose more than 10% to leakage, lint or schema
    fails = []
    for name in NEW_POOLS:
        kept = Counter(r["meta"]["kind"] for r in pools[name])
        for kind, n in kind_in[name].items():
            if n and kept[kind] / n < 0.9:
                fails.append((name, kind, kept[kind], n))
    if fails:
        raise SystemExit("FAIL: kinds keep < 90% of their rows: " + str(fails[:10]))
    for src, ks in kind_leak.items():
        for kind, n in ks.items():
            tot = kind_in[src][kind]
            if tot and (tot - n) / tot < 0.5 and src not in NEW_POOLS:
                fails.append((src, kind, tot - n, tot))
            elif tot and (tot - n) / tot < 0.9 and src not in NEW_POOLS:
                print(f"note: legacy {src}/{kind} keeps {tot - n}/{tot} after exact-text leakage drops (real overlap, see report)")
    if fails:
        raise SystemExit("FAIL: legacy kinds lose > 10% to leakage: " + str(fails[:10]))

    # ---- split: val per source, before repetition
    train_pool, val = {}, []
    for src, rows in pools.items():
        rng.shuffle(rows)
        n = len(rows)
        target = max(5, int(round(n * a.val_frac))) if n >= 40 else 0
        flagged = [r for r in rows if val_flag.get(id(r))]
        rest = [r for r in rows if not val_flag.get(id(r))]
        v = flagged[:target]
        if len(v) < target:
            v += rest[:target - len(v)]
        vset = {id(r) for r in v}
        train_pool[src] = [r for r in rows if id(r) not in vset]
        val += v

    # ---- repetition and restraint top-up
    train = []
    comp = Counter()
    for src, rows in train_pool.items():
        k = rep.get(src, 1)
        for _ in range(k):
            train += rows
        comp[src] = len(rows) * k
    R = sum(is_restraint(r) for r in train)
    T = len(train)
    topup = 0
    if R / T < old_share:
        x = math.ceil((old_share * T - R) / (1 - old_share))
        # top up from the legacy teacher answers (plain knowledge answers with the tool block present), without replacement
        cand = [r for r in train_pool.get("restraint_teacher", []) if is_restraint(r)]
        rng.shuffle(cand)
        extra = []
        while len(extra) < x and cand:
            extra += cand[: x - len(extra)]
            if len(extra) >= 2 * len(cand):                      # never more than x3 of one row
                break
        train += extra
        topup = len(extra)
        for r in extra:
            comp[r["meta"]["source"]] += 1
        R += len(extra); T += len(extra)

    # ---- token cap on the final rows (Qwen3 tokenizer, 19-tool block)
    tok = None
    if not a.no_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(LAB, "..", "base", "Qwen3-4B-Instruct-2507"))
    lens = []

    def ntok(row):
        text = F.render_text(row["messages"], row["tools"], "4b", bio=row["meta"]["bio"])
        return len(tok(text, add_special_tokens=False)["input_ids"]) if tok else len(text) // 3

    cache = {}

    def keep(row):
        k = id(row)
        if k not in cache:
            n = ntok(row)
            cache[k] = n
        return cache[k] <= MAX_TOKENS

    def roundtrip(row):
        for m in row["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                calls, bad = F.parse_calls(F.assistant_text(m))
                if bad or [{"name": c["name"], "arguments": c["arguments"]} for c in m["tool_calls"]] != calls:
                    return False
        return True

    def filt(rows, label):
        out = []
        for r in rows:
            if not roundtrip(r):
                stats["schema_drop"][(r["meta"]["source"], "round trip")] += 1
                continue
            if not keep(r):
                stats["token_drop"][r["meta"]["source"]] += 1
                continue
            out.append(r)
        return out
    train = filt(train, "train")
    val = filt(val, "val")
    lens = [cache[id(r)] for r in train]

    rng.shuffle(train)
    rng.shuffle(val)
    for path, rows in ((a.out_prefix + "_train.jsonl", train), (a.out_prefix + "_val.jsonl", val)):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report
    R2 = sum(is_restraint(r) for r in train)
    cnt_t, cnt_v = Counter(r["meta"]["source"] for r in train), Counter(r["meta"]["source"] for r in val)
    new_rows = sum(cnt_t[s] for s in NEW_POOLS)
    print(f"\ntrain={len(train)} val={len(val)}  (new pools {new_rows} = {new_rows / len(train):.1%} of train)")
    print(f"tool-restraint share: sft3 {old_share:.4f} -> sft4 {R2 / len(train):.4f} ({R2} rows, top-up {topup})")
    print(f"{'source':20} {'unique':>7} {'train':>7} {'val':>5}")
    for s in NEW_POOLS + ["core_math", "core_reads", "core_trade", "restraint_teacher", "restraint_code", "restraint_strategy", "restraint_voice", "restraint_misc"]:
        print(f"{s:20} {len(pools.get(s, [])):>7} {cnt_t[s]:>7} {cnt_v[s]:>5}")
    print(f"{'TOTAL':20} {sum(len(v) for v in pools.values()):>7} {len(train):>7} {len(val):>5}")
    bios = Counter(r["meta"]["bio"] for r in train)
    print("device word (train):", dict(bios))
    if lens:
        import statistics
        ls = sorted(lens)
        print(f"tokens{'' if tok else ' (chars/3 estimate)'}: mean {statistics.mean(ls):.0f} p50 {ls[len(ls) // 2]} p90 {ls[int(len(ls) * .9)]} max {ls[-1]}  total ~{sum(ls):,}")
        blk = len(tok(F.tools_block("Face ID"), add_special_tokens=False)["input_ids"]) if tok else 0
        over = {n: sum(1 for x in ls if x > n) for n in (3200, 4096)}
        print(f"19-tool block = {blk} tokens. Rows longer than 3200 tokens (the h12-4b-gpu2 max_len): {over[3200]} ({over[3200] / len(ls):.0%}); "
              f"longer than 4096: {over[4096]}. The trainer needs max_len >= {ls[-1]} (rows over it are dropped by sft_cpu.encode).")
    print("\ndrops")
    for k, v in stats.items():
        if v:
            print(f"  {k}: {sum(v.values())}")
            for kk, vv in v.most_common(12):
                print(f"      {vv:5} {kk}")
    kinds_t = Counter((r["meta"]["source"], r["meta"]["kind"]) for r in train)
    report = {"train": len(train), "val": len(val), "by_source_train": dict(cnt_t), "by_source_val": dict(cnt_v),
              "unique_by_source": {k: len(v) for k, v in pools.items()}, "restraint_share_old": old_share, "restraint_share_new": R2 / len(train),
              "topup": topup, "drops": {k: {str(kk): vv for kk, vv in v.items()} for k, v in stats.items()},
              "by_kind_train": {f"{s}/{k}": v for (s, k), v in sorted(kinds_t.items())}, "repeat": rep,
              "leak_index": {"eval_items": n_ev, "distinct_user_texts": len(ix.exact), "long_texts": ix.n}}
    json.dump(report, open(a.out_prefix + "_report.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
