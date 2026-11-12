#!/usr/bin/env python3
"""Assemble the sft5 training and validation sets (tools v3, packs): data5/sft5_train.jsonl, data5/sft5_val.jsonl,
data5/STATS.md (+ data5/sft5_report.json).

Inputs (read only): data3/sft4_train.jsonl, data3/sft4_val.jsonl (the v2 set) and the data4/<generator>.jsonl pools.
Output rows: {"messages": [...], "tools": bool, "meta": {...}}. Render with gen4/fmt4.py:
    fmt4.render(row["messages"], row["meta"]["pack"], "4b")                 # Hermit 4B
    fmt4.render(row["messages"], row["meta"]["pack_lite"], "lite")          # Hermit Lite, only rows with "lite" in family_ok
`meta.pack` / `meta.pack_lite` are tool-name lists in registry order ([] = chat mode, no tool block).

What it does (PLAN_v3.md section 0.7, section 4, C2, C10, C13)
  1. sft4 rows. Dropped as now wrong: kinds neg_sell, neg_cancel_order; every row whose assistant text matches the PLAN 0.7
     regex or a refusal of something a v3 tool does (STALE below: automations/recurring, positions/portfolio, imperative
     navigation such as "go to Trade > Positions"). Rewritten: neg_nav and not_activity_howto ("tap Earn in the tab bar",
     "it's in Wallets > Activity") become an open_page call + "I put a button below ..." answer. Tools rows get
     SYSTEM_AGENT_v3 with the row's device word; the name / hermit.md suffixes are kept byte for byte. Chat rows untouched.
  2. data4 rows: kept as written (system already v3); meta slimmed (world states dropped, origin file:line kept).
  3. Leakage: every user turn of eval2, eval3 and eval4 (contexts and scripts); a hit drops the training row.
  4. De-duplication across all sources: key = (system text after the base prompt, normalised user turns). First seen wins in
     the order data4 (generator order) -> sft4 train -> sft4 val. Kept rows are distinct; the only repeated rows are the
     restraint top-up of step 7 (identical copies, never two golds for one prompt).
  5. Packs. data4 rows keep the generator's pack when it is a pack the app can send (packs.is_valid) and holds every call.
     Every other tools row: 15% (stable hash) all 30 tools (PLAN 0.3: 15% of 4B rows see the full block, as on Private
     Cloud), else packs.choose(first user message, attachment, "4b"); when the row calls a tool outside it, the smallest
     router pack that holds all its calls (what the app sends after the user reopens the chat from the open_page button).
  6. family_ok (C2): "lite" when the row has at most 3 tool calls, a conversation under 2,048 tokens after the system/tool
     prefix (Lite render), and its calls fit a Lite pack (core + one pack, Lite trade pack, 16 tools at most). pack_lite =
     the row's pack if it is a Lite pack, else the Lite router choice if it holds the calls, else the smallest Lite pack
     holding them. Rows that say a tool "isn't available in this chat" need pack_lite == pack.
  7. Restraint share (tools present, no tool call) kept near the sft4 share: top-up with sft4 restraint rows (at most x3).
  8. Split: sft4_val rows stay val; data4 sources with 40+ rows give 2% (min 5) to val. Shuffle (seed 5), write, STATS.

    python gen4/assemble5.py              # full run with the Qwen3 tokenizers
    python gen4/assemble5.py --no-tokens  # chars/3.4 estimate (fast iteration)
"""
import argparse, hashlib, json, math, os, random, re, statistics, sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import fmt4 as F                                   # noqa: E402
import packs as P                                  # noqa: E402
sys.path.insert(1, os.path.join(LAB, "gen3"))
import assemble4 as A4                             # noqa: E402  (gen3, read only: LeakIndex, norm, build_leak_index)

D3, D4, D5 = (os.path.join(LAB, d) for d in ("data3", "data4", "data5"))
EV4 = os.path.join(LAB, "eval4")
DATA4 = ["spot_solana", "orders_protect", "contacts_pay", "auto_pages", "portfolio_caps", "multistep", "notes_name",
         "recovery_errors", "followups_stale", "rounding_brief", "ask_wording", "lite_cloud", "drift_fix"]
EXTRA4 = ["safety_mixed"]                          # optional pools in data5/pools/ (gen4/safety_mixed.py): used when the file exists
ALL30_SHARE = 0.15
LITE_CTX_TOKENS = 2048
LITE_MAX_STEPS = 3
CAP_4B, CAP_LITE = 8192, 6144
RESTRAINT_SOURCES = ("restraint_teacher", "restraint_code", "restraint_strategy", "restraint_voice", "restraint_misc")

# ------------------------------------------------------------------ stale rows (PLAN_v3 0.7 and the refusal search)
DROP_KINDS = {"neg_sell": "PLAN 0.7: 'I can only prepare buys' (prepare_sell exists)",
              "neg_cancel_order": "PLAN 0.7: 'I can't cancel orders' (cancel_order exists)"}
REWRITE_KINDS = {"neg_nav", "not_activity_howto"}
STALE = [
    ("plan_0.7", re.compile(r"use Swap in Wallets|can only prepare buys|can't cancel|cannot cancel|tab bar|Tap Earn|tap Earn")),
    ("no_automation", re.compile(r"(does ?n[o']t|does not|doesn't|\bno\b|\bnot\b) (support|have|offer)[^.\n]{0,40}"
                                 r"(recurring|automat|schedul|auto-?trad|\bbots?\b|alerts?\b)|no built-in (scheduler|automation)"
                                 r"|(can'?t|cannot) (automate|set up (an? )?(automation|alert|reminder|recurring))", re.I)),
    ("no_positions", re.compile(r"(can'?t|cannot|unable to) (see|show|read|check) your (positions?|portfolio|holdings)", re.I)),
    ("no_contacts", re.compile(r"(can'?t|cannot|no way to|doesn'?t)[^.\n]{0,30}(save|store|add)[^.\n]{0,20}(contact|address book|payee)", re.I)),
    ("no_token_search", re.compile(r"(can'?t|cannot)[^.\n]{0,20}(search|look up)[^.\n]{0,20}\b(tokens?|mints?)\b", re.I)),
    ("no_stop_edit", re.compile(r"(can'?t|cannot)[^.\n]{0,30}\b(add|set|move|change)\b[^.\n]{0,20}(stop|take profit)", re.I)),
    ("nav_imperative", re.compile(r"\b(go to|head to|navigate to|open|check your)\s+(the\s+)?(Trade|Wallets?|Earn|Settings|Payments|Explore|Home)"
                                  r"\s?>\s?\w+|\b(go to|head to|open)\s+(the\s+)?(Trade|Wallets?|Earn|Settings|Payments|Explore)"
                                  r"\b[^.\n]{0,30}?,?\s+(then|and then|and tap|then tap)\b", re.I)),
]
CHAT_EXEMPT = {"no_positions", "nav_imperative"}  # chat mode has no tools: "I can't see your wallets from chat" is right there

PAGE_WORDS = [("wallets", re.compile(r"wallet", re.I), "Wallets"), ("earn", re.compile(r"\bearn\b", re.I), "Earn"),
              ("trade", re.compile(r"\btrade\b", re.I), "Trade"), ("payments", re.compile(r"payment", re.I), "Payments"),
              ("settings", re.compile(r"setting", re.I), "Settings")]
NAV_ANSWERS = ["I put a button below that opens {P}.", "I put a button below. Tap it to open {P}.",
               "I put a button below for {P}.", "I put a button below; it takes you to {P}.",
               "I put a button below that takes you to {P}.", "I put a button below that opens the {P} page."]
ACT_ANSWERS = ["I put a button below that opens Wallets, where Activity lists each send, swap and buy with its status. I can also read your recent activity here.",
               "I put a button below for Wallets. Activity there shows each item with its status: confirmed, failed or in progress. Want me to check it here?",
               "I put a button below that opens Wallets; your activity is listed there with each item's status. I can also pull it up in this chat.",
               "I put a button below that takes you to Wallets, where Activity shows your sends, swaps and buys. Should I show your recent activity here?",
               "I put a button below for Wallets. Activity there lists what you've sent, swapped and bought, with the status of each. I can read it out here too.",
               "I put a button below that opens Wallets. Activity there has every send, swap and buy with its status. Want me to show it here?"]


def stale_hits(row):
    out = []
    for m in row["messages"]:
        if m["role"] != "assistant" or not m.get("content"):
            continue
        for name, rx in STALE:
            if not row["tools"] and name in CHAT_EXEMPT:
                continue
            if rx.search(m["content"]):
                out.append(name)
    return out


def rewrite_nav(row, rng):
    """neg_nav / not_activity_howto -> open_page call + button answer. Returns the new row or None."""
    msgs = row["messages"]
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if len(users) != 1 or len(msgs) != users[0] + 2:
        return None
    u = msgs[users[0]]["content"]
    if row["meta"]["kind"] == "not_activity_howto":
        page, label, ans = "wallets", "Wallets", rng.choice(ACT_ANSWERS)
    else:
        found = [(p, lab) for p, rx, lab in PAGE_WORDS if rx.search(u)]
        if len(found) != 1:
            return None
        page, label = found[0]
        ans = rng.choice(NAV_ANSWERS).replace("{P}", label)
    res = {"note": f"The user taps it to open {label}.", "page": page, "screen": page, "status": "button_shown"}
    new = msgs[:users[0] + 1] + [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "open_page", "arguments": {"page": page}}]},
        {"role": "tool", "content": json.dumps(res, ensure_ascii=False)},
        {"role": "assistant", "content": ans}]
    out = {"messages": new, "tools": True, "meta": dict(row["meta"])}
    out["meta"]["kind"] = row["meta"]["kind"] + "_v3"
    out["meta"]["rewritten"] = "open_page"
    return out


def to_v3_system(row):
    """Swap the v2 agent prompt for SYSTEM_AGENT_v3 (same device word), keeping the name / notes suffix."""
    bio = row["meta"]["bio"]
    s = row["messages"][0]["content"]
    base = F.system_agent_v2(bio)
    if not s.startswith(base):
        raise ValueError("tools row without the v2 agent prompt")
    row["messages"][0] = dict(row["messages"][0], content=F.system_agent(bio) + s[len(base):])
    return row


# ------------------------------------------------------------------ helpers
def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                out.append((i + 1, json.loads(line)))
    return out


def calls_of(row):
    return [tc["name"] for m in row["messages"] for tc in (m.get("tool_calls") or [])]


ATT_RE = re.compile(r"^The user attached the file (.+?)(?: \([^\n]*\))?:\n<file>\n", re.S)


def first_user(row):
    """(typed text without file bodies, attachment file name or False) of the first user message."""
    for m in row["messages"]:
        if m["role"] == "user":
            t = m["content"]
            mm = ATT_RE.match(t)
            att = mm.group(1) if mm else False
            return without_files(t), att
    return "", False


def without_files(s):
    while "<file>" in s:
        a = s.index("<file>")
        b = s.find("</file>", a + 6)
        s = s[:a] if b < 0 else s[:a] + s[b + 7:]
    return s


def sys_suffix(row):
    s = row["messages"][0]["content"] if row["messages"] and row["messages"][0]["role"] == "system" else ""
    bio = row["meta"].get("bio") or F.detect_bio(row["messages"])
    for base in (F.system_agent(bio), F.SYSTEM_CHAT):
        if s.startswith(base):
            return s[len(base):]
    return s


def dedup_key(row):
    us = [A4.norm(m["content"]) for m in row["messages"] if m["role"] == "user"]
    return (A4.norm(sys_suffix(row)), "|".join(us))


def u01(*parts):
    h = hashlib.sha1("\x1f".join(parts).encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def is_restraint(row):
    return bool(row["meta"].get("pack")) and not calls_of(row)


NOT_AVAIL = re.compile(r"(isn'?t|is not|aren'?t|are not|not) available in this chat", re.I)


# ------------------------------------------------------------------ tokens
class Tok:
    def __init__(self, use):
        self.use = use
        self.t4 = self.tl = None
        if use:
            from transformers import AutoTokenizer
            self.t4 = AutoTokenizer.from_pretrained(os.path.join(LAB, "..", "base", "Qwen3-4B-Instruct-2507"))
            self.tl = AutoTokenizer.from_pretrained(os.path.join(LAB, "..", "base", "Qwen3-1.7B"))
        self.cache = {}

    def n(self, text, fam):
        if not self.use:
            return int(len(text) / 3.4) + 1
        tk = self.t4 if fam == "4b" else self.tl
        return len(tk(text, add_special_tokens=False)["input_ids"])

    def prefix(self, sys_text, pack, bio, fam):
        k = (sys_text, tuple(pack), bio, fam)
        if k not in self.cache:
            self.cache[k] = self.n(F.system_prefix(sys_text, pack, bio), fam)
        return self.cache[k]

    def lengths(self, row, pack, fam):
        """(total tokens, conversation tokens after the system/tool prefix). Additive: every segment starts at <|im_start|>."""
        segs = F.render(row["messages"], pack, fam, bio=row["meta"]["bio"])
        rest = "".join(t for t, _ in segs[1:])
        sys_text = row["messages"][0]["content"] if row["messages"][0]["role"] == "system" else ""
        pre = self.prefix(sys_text, pack, row["meta"]["bio"], fam)
        r = self.n(rest, fam)
        return pre + r, r


# ------------------------------------------------------------------ leakage index (eval2 + eval3 + eval4)
def leak_index():
    ix, n = A4.build_leak_index()
    files = [(EV4, f) for f in sorted(os.listdir(EV4)) if f.endswith(".jsonl")]
    files += [(os.path.join(LAB, "eval5"), p + ".jsonl") for p in EXTRA4 if os.path.exists(os.path.join(LAB, "eval5", p + ".jsonl"))]
    for d, f in files:
        for _, r in load(os.path.join(d, f)):
            n += 1
            users = [m["content"] for m in r.get("context", []) or [] if m["role"] == "user"]
            for u in users:
                ix.add(u, os.path.basename(d) + "/" + f)
            ix.add_seq(users)
            script = [s if isinstance(s, str) else (s.get("content") or s.get("text") or "") for s in (r.get("script") or [])]
            for s in script:
                ix.add(s, os.path.basename(d) + "/" + f)
            if script:
                ix.add_seq(users + script)
    return ix, n


# ------------------------------------------------------------------ main
def slim4(r, pool, line):
    m = dict(r["meta"])
    for k in ("world", "world_final", "world_last_user", "world_start", "est_tokens"):
        m.pop(k, None)
    m["source"] = pool
    m["generator"] = m.get("generator") or pool
    m["origin"] = f"data4/{pool}.jsonl:{line}"
    m["bio"] = m.get("bio") or F.detect_bio(r["messages"])
    m["gen_pack"] = r["meta"].get("pack")
    m["gen_family_ok"] = r["meta"].get("family_ok")
    m.pop("pack", None)
    m.pop("family_ok", None)
    return {"messages": r["messages"], "tools": bool(r["tools"]), "meta": m}


def assign_pack(row, st):
    """meta.pack (4B) and meta.pack_source."""
    m = row["meta"]
    if not row["tools"]:
        m["pack"], m["pack_name"], m["pack_source"] = [], "chat", "chat"
        return
    calls = set(calls_of(row))
    gp = m.get("gen_pack")
    if gp:
        gp = F.norm_pack(gp)
        if (P.is_valid(gp, "4b") or P.is_valid(gp, "lite")) and calls <= set(gp):
            # a Lite-only pack (core + the Lite trade pack) is kept for 4B too: the rendered block is what the gold assumes
            m["pack"], m["pack_source"] = gp, "generator"
            m["pack_name"] = P.label(gp)
            st["pack_source"]["generator"] += 1
            return
        st["generator_pack_replaced"][(m["source"], len(gp), "calls outside" if not calls <= set(gp) else "not a router pack")] += 1
    text, att = first_user(row)
    key = "|".join(A4.norm(x["content"]) for x in row["messages"] if x["role"] == "user")
    if u01("all30", key) < ALL30_SHARE:
        pack, src = list(P.ALL30), "all30"
    else:
        pack = P.choose(text, att, "4b")
        src = "router"
        if not calls <= set(pack):
            pack, src = P.smallest_containing(calls, "4b", prefer=pack), "router_fallback"
            st["fallback"][(m["source"], P.name_of(pack, "4b"))] += 1
    m["pack"], m["pack_source"], m["pack_name"] = pack, src, P.label(pack)
    st["pack_source"][src] += 1


def assign_lite(row, tok, st):
    m = row["meta"]
    calls = calls_of(row)
    fam = ["4b"]
    m["pack_lite"] = None
    if not row["tools"]:
        pl = []
    else:
        pack = m["pack"]
        need = set(calls)
        if P.is_valid(pack, "lite"):
            pl = pack
        else:
            text, att = first_user(row)
            ch = P.choose(text, att, "lite")
            pl = ch if need <= set(ch) else P.smallest_containing(need, "lite", prefer=ch)
        if pl is not None and any(NOT_AVAIL.search(x.get("content") or "") for x in row["messages"] if x["role"] == "assistant"):
            if set(pl) != set(pack):
                pl = None
                st["lite_reject"]["says not available, Lite pack differs"] += 1
    reason = None
    if pl is None:
        reason = "calls fit no Lite pack"
    elif len(calls) > LITE_MAX_STEPS:
        reason = "more than 3 steps"
    else:
        tot, ctx = tok.lengths(row, pl, "lite")
        m["tokens_lite"] = tot
        if ctx >= LITE_CTX_TOKENS:
            reason = "conversation over 2k tokens"
        elif tot > CAP_LITE:
            reason = "over the Lite cap"
    if reason is None:
        fam.append("lite")
        m["pack_lite"] = pl
        m["pack_lite_name"] = P.label(pl)
    else:
        st["lite_reject"][reason] += 1
        m.pop("tokens_lite", None)
    m["family_ok"] = fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=D5)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    nav_rng = random.Random(55)
    st = defaultdict(Counter)

    # ---------------- v2 shares (sft4 as trained), same definition as for sft5: tools present and no tool call
    sft4_train = load(os.path.join(D3, "sft4_train.jsonl"))
    sft4_val = load(os.path.join(D3, "sft4_val.jsonl"))
    v2_restraint = sum(1 for _, r in sft4_train if r["tools"] and not calls_of(r)) / len(sft4_train)
    print(f"sft4 train {len(sft4_train)} rows, val {len(sft4_val)}; restraint share (tools, no call) {v2_restraint:.4f}")

    ix, n_ev = leak_index()
    print(f"leak index: {n_ev} eval items (eval2, eval3, eval4, eval5 extra pools), {len(ix.exact)} distinct user texts")

    candidates = []                      # (row, split_hint) in priority order
    # ---------------- data4
    pools_in = Counter()
    for pool in DATA4 + EXTRA4:
        path = os.path.join(D4 if pool in DATA4 else os.path.join(D5, "pools"), pool + ".jsonl")
        rows = load(path)
        if not rows:
            if pool in DATA4:
                raise SystemExit(f"missing pool {path}")
            continue
        for line, r in rows:
            pools_in[pool] += 1
            candidates.append((slim4(r, pool, line), "data4"))
    # ---------------- sft4
    for split, rows in (("train", sft4_train), ("val", sft4_val)):
        for line, r in rows:
            m = r["meta"]
            st["sft4_in"][split] += 1
            kind = m.get("kind")
            if kind in DROP_KINDS:
                st["dropped_stale"][f"kind {kind}"] += 1
                continue
            r = json.loads(json.dumps(r))
            r["meta"]["origin"] = f"data3/sft4_{split}.jsonl:{line}"
            if kind in REWRITE_KINDS:
                nr = rewrite_nav(r, nav_rng)
                if nr is None:
                    st["dropped_stale"][f"kind {kind} (no page match)"] += 1
                    continue
                r = nr
                st["rewritten"][kind] += 1
            else:
                h = stale_hits(r)
                if h:
                    st["dropped_stale"][h[0]] += 1
                    st["stale_by_source"][(m["source"], m.get("kind"), h[0])] += 1
                    continue
            if r["tools"]:
                try:
                    to_v3_system(r)
                except ValueError as e:
                    st["dropped_other"][str(e)] += 1
                    continue
            r["meta"]["tools_version"] = "v3" if r["tools"] else None
            candidates.append((r, "sft4_" + split))

    # ---------------- leakage + dedup
    kept, seen = [], {}
    kind_in, kind_kept = Counter(), Counter()
    for row, hint in candidates:
        src, kind = row["meta"]["source"], row["meta"].get("kind")
        kind_in[(src, kind)] += 1
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        if hint == "data4":
            leak = ix.hit_row(users, fuzzy=True)
        else:
            leak = ix.hit_row(users, fuzzy=False)
        if leak:
            st["leak_drop"][(src, leak)] += 1
            continue
        k = dedup_key(row)
        if k in seen:
            if seen[k] == src and hint != "data4" and src in RESTRAINT_SOURCES + ("restraint_misc",):
                st["dup_repeat_collapsed"][src] += 1          # sft4's deliberate repeats (voice x4, code x3, ...)
            else:
                st["dup_drop"][(src, "same as " + seen[k])] += 1
            continue
        seen[k] = src
        kind_kept[(src, kind)] += 1
        kept.append((row, hint))

    fails = [(s, k, kind_kept[(s, k)], n) for (s, k), n in kind_in.items()
             if s in DATA4 + EXTRA4 and n >= 20 and kind_kept[(s, k)] / n < 0.9]
    small = [(s, k, kind_kept[(s, k)], n) for (s, k), n in kind_in.items()
             if s in DATA4 + EXTRA4 and n < 20 and kind_kept[(s, k)] / n < 0.9]
    st["small_kinds_under_90"] = Counter({f"{s}/{k} {kk}/{n}": 1 for s, k, kk, n in small})
    if small:
        print("note: kinds under 20 rows that keep < 90% (2 or fewer rows lost to leakage):", small)
    if fails:
        print("FAIL: data4 kinds keeping < 90% after leakage and dedup:", fails[:20])
        raise SystemExit(1)

    # ---------------- packs, family, tokens
    tok = Tok(not a.no_tokens)
    for row, _ in kept:
        assign_pack(row, st)
    for row, _ in kept:
        assign_lite(row, tok, st)
        pk = row["meta"]["pack"]
        row["meta"]["tokens_4b"] = tok.lengths(row, pk, "4b")[0]
        if row["meta"]["tokens_4b"] > CAP_4B:
            st["token_drop"][row["meta"]["source"]] += 1
            row["meta"]["_drop"] = True
    kept = [(r, h) for r, h in kept if not r["meta"].pop("_drop", False)]

    # ---------------- check4 (PLAN_v3 C9) as the last filter: a row failing any check is dropped and counted here; the
    # standalone `python gen4/check4.py` then re-checks the written files and must print OK.
    import check4 as C4
    gen_sources = set(DATA4 + EXTRA4)
    out = []
    for row, hint in kept:
        errs = C4.check_row(row, None, gen_sources)
        if errs:
            for name, msg in errs[:1]:
                st["check4_drop"][(row["meta"]["source"], name, re.sub(r"[\d.,$]+", "#", msg)[:70])] += 1
            continue
        out.append((row, hint))
    if len(kept) - len(out) > 0.01 * len(kept):
        raise SystemExit(f"FAIL: check4 drops {len(kept) - len(out)} rows (> 1%): {st['check4_drop'].most_common(10)}")
    kept = out

    # ---------------- split
    train, val = [], []
    by_src = defaultdict(list)
    for row, hint in kept:
        if hint == "sft4_val":
            val.append(row)
        elif hint == "data4":
            by_src[row["meta"]["source"]].append(row)
        else:
            train.append(row)
    for src in sorted(by_src):
        rows = by_src[src]
        rng.shuffle(rows)
        nv = max(5, int(round(len(rows) * a.val_frac))) if len(rows) >= 40 else 0
        val += rows[:nv]
        train += rows[nv:]

    # ---------------- restraint top-up (identical copies of sft4 restraint rows, at most 3 copies of one row)
    R = sum(is_restraint(r) for r in train)
    T = len(train)
    share0 = R / T
    topup = []
    if share0 < v2_restraint - 0.005:
        need = math.ceil((v2_restraint * T - R) / (1 - v2_restraint))
        cand = [r for r in train if is_restraint(r) and r["meta"]["source"] in RESTRAINT_SOURCES]
        rng.shuffle(cand)
        while len(topup) < need and cand and len(topup) < 2 * len(cand):
            topup += cand[: need - len(topup)]
        for r in topup:
            st["topup"][r["meta"]["source"]] += 1
        train += [json.loads(json.dumps(r)) for r in topup]
    share1 = sum(is_restraint(r) for r in train) / len(train)
    vs = sum(is_restraint(r) for r in val) / max(1, len(val))

    rng.shuffle(train)
    rng.shuffle(val)
    for name, rows in (("sft5_train.jsonl", train), ("sft5_val.jsonl", val)):
        with open(os.path.join(a.out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_stats(a, train, val, st, v2_restraint, share0, share1, vs, pools_in, tok, len(topup), n_ev)
    print(f"train {len(train)}  val {len(val)}  restraint share v2 {v2_restraint:.4f} -> sft5 {share1:.4f} (before top-up {share0:.4f}, top-up {len(topup)})")
    return 0


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def dist(xs):
    if not xs:
        return "-"
    s = sorted(xs)
    return f"mean {statistics.mean(s):,.0f}, p50 {s[len(s) // 2]:,}, p90 {s[int(len(s) * .9)]:,}, p99 {s[int(len(s) * .99)]:,}, max {s[-1]:,}"


def write_stats(a, train, val, st, v2_share, share0, share1, vshare, pools_in, tok, n_topup, n_ev):
    L = []
    w = L.append
    allrows = train + val
    w("# data5 / sft5: statistics\n")
    w(f"Written by `gen4/assemble5.py` ({'Qwen3 tokenizers: Qwen3-4B-Instruct-2507 for 4B, Qwen3-1.7B for Lite' if tok.use else 'token counts are chars/3.4 ESTIMATES (--no-tokens)'}). "
      f"Render: `fmt4.render(messages, meta.pack, \"4b\")`; Lite: `fmt4.render(messages, meta.pack_lite, \"lite\")` on rows with `\"lite\"` in `meta.family_ok`.\n")
    w(f"**train {len(train):,} rows, val {len(val):,} rows.** Unique training rows {len(train) - n_topup:,} + {n_topup:,} restraint top-up copies.\n")
    w("## Rows per source\n")
    w("| source | in | train | val | lite-ok (train) |\n|---|---:|---:|---:|---:|")
    ct, cv = Counter(r["meta"]["source"] for r in train), Counter(r["meta"]["source"] for r in val)
    cl = Counter(r["meta"]["source"] for r in train if "lite" in r["meta"]["family_ok"])
    sin = Counter()
    for k, v in pools_in.items():
        sin[k] = v
    srcs = sorted(set(ct) | set(cv), key=lambda s: (s not in pools_in, s))
    for s in srcs:
        w(f"| {s} | {sin.get(s, '') or ''} | {ct[s]:,} | {cv[s]:,} | {cl[s]:,} |")
    w(f"| **total** | | **{len(train):,}** | **{len(val):,}** | **{sum(cl.values()):,}** |\n")
    d4 = sum(ct[s] for s in pools_in)
    w(f"data4 rows in train: {d4:,} ({pct(d4, len(train))}); sft4 rows: {len(train) - d4:,}.\n")

    w("## sft4 rows dropped or rewritten as now wrong (PLAN_v3 0.7 + refusal search)\n")
    w(f"sft4 rows in: {st['sft4_in']['train']:,} train + {st['sft4_in']['val']:,} val.\n")
    w("| reason | rows |\n|---|---:|")
    for k, v in st["dropped_stale"].most_common():
        w(f"| dropped: {k} | {v} |")
    w(f"| **dropped total** | **{sum(st['dropped_stale'].values())}** |")
    for k, v in st["rewritten"].most_common():
        w(f"| rewritten to open_page: {k} | {v} |")
    w("")
    if st["stale_by_source"]:
        w("Refusal-search drops by source/kind: " + ", ".join(f"{s}/{k} {r} x{v}" for (s, k, r), v in st["stale_by_source"].most_common()) + ".\n")
    w("## Leakage and de-duplication\n")
    w(f"Leak index: {n_ev:,} eval items (eval2, eval3, eval4 and eval5/safety_mixed; contexts and scripts).\n")
    w(f"- leakage drops: {sum(st['leak_drop'].values())} " + ", ".join(f"{s}/{r} {v}" for (s, r), v in st["leak_drop"].most_common(20)))
    w(f"- duplicate user texts dropped: {sum(st['dup_drop'].values())} " + ", ".join(f"{s} ({o}) {v}" for (s, o), v in st["dup_drop"].most_common(20)))
    w(f"- sft4's deliberate repeats collapsed: {sum(st['dup_repeat_collapsed'].values())} " + ", ".join(f"{s} {v}" for s, v in st["dup_repeat_collapsed"].most_common()))
    w(f"- over the 4B cap ({CAP_4B}): {sum(st['token_drop'].values())}")
    w(f"- failing gen4/check4.py (PLAN_v3 C9), dropped: {sum(st['check4_drop'].values())} " +
      "; ".join(f"{s}: {c} ({m}) {v}" for (s, c, m), v in st["check4_drop"].most_common(20)) + "\n")
    w("## Packs (4B, `meta.pack`)\n")
    w("| pack | tools | train | val |\n|---|---:|---:|---:|")
    pt, pv = Counter(r["meta"]["pack_name"] for r in train), Counter(r["meta"]["pack_name"] for r in val)
    ntools = {r["meta"]["pack_name"]: len(r["meta"]["pack"]) for r in allrows}
    for k, v in sorted(pt.items(), key=lambda x: -x[1]):
        w(f"| {k} | {ntools[k]} | {v:,} ({pct(v, len(train))}) | {pv[k]:,} |")
    w("")
    w("Where the pack came from (train+val): " + ", ".join(f"{k} {v:,}" for k, v in st["pack_source"].most_common()) +
      ". `router_fallback` = the row calls a tool outside the router's choice and gets the smallest router pack holding its calls: "
      f"{sum(st['fallback'].values())} rows (" + ", ".join(f"{s}->{p} {v}" for (s, p), v in st["fallback"].most_common(12)) + ").\n")
    if st["generator_pack_replaced"]:
        w("Generator packs replaced (not a pack the app can send, or calls outside it): " +
          ", ".join(f"{s} ({n} tools, {why}) {v}" for (s, n, why), v in st["generator_pack_replaced"].most_common()) + ".\n")
    w("Packs per source (train):\n")
    w("| source | " + " | ".join(sorted(pt)) + " |\n|---|" + "---:|" * len(pt))
    for s in srcs:
        c = Counter(r["meta"]["pack_name"] for r in train if r["meta"]["source"] == s)
        w(f"| {s} | " + " | ".join(str(c[k]) if c[k] else "" for k in sorted(pt)) + " |")
    w("")
    w("## family_ok (PLAN_v3 C2)\n")
    fo = Counter(tuple(r["meta"]["family_ok"]) for r in train)
    fov = Counter(tuple(r["meta"]["family_ok"]) for r in val)
    w(f"- train: 4B only {fo[('4b',)]:,}, 4B + Lite {fo[('4b', 'lite')]:,}; val: 4B only {fov[('4b',)]:,}, 4B + Lite {fov[('4b', 'lite')]:,}.")
    w("- not Lite because: " + ", ".join(f"{k} {v:,}" for k, v in st["lite_reject"].most_common()) + " (train+val).")
    plc = Counter(r["meta"].get("pack_lite_name") for r in train if "lite" in r["meta"]["family_ok"])
    w("- Lite packs (`meta.pack_lite`, train): " + ", ".join(f"{k} {v:,}" for k, v in plc.most_common()) + ".")
    agree = Counter()
    for r in allrows:
        g = r["meta"].get("gen_family_ok")
        if g is not None:
            agree["same" if sorted(g) == sorted(r["meta"]["family_ok"]) else f"generator {'+'.join(g)} -> {'+'.join(r['meta']['family_ok'])}"] += 1
    if agree:
        w("- data4 rows vs the generator's own family_ok: " + ", ".join(f"{k} {v:,}" for k, v in agree.most_common()) + ".\n")
    w("## Restraint share (tools present, no tool call)\n")
    w(f"sft4 (v2) {v2_share:.4f}; sft5 before top-up {share0:.4f}; after top-up {share1:.4f} ({n_topup} copies: " +
      ", ".join(f"{k} {v}" for k, v in st["topup"].most_common()) + f"); val {vshare:.4f}.\n")
    w("## Token lengths (with the pack prefix)\n")
    l4 = [r["meta"]["tokens_4b"] for r in train]
    ll = [r["meta"]["tokens_lite"] for r in train if "lite" in r["meta"]["family_ok"]]
    w(f"- Hermit 4B, all train rows ({len(l4):,}): {dist(l4)}. Cap {CAP_4B}.")
    w(f"- Hermit Lite, lite-ok train rows ({len(ll):,}): {dist(ll)}. Cap {CAP_LITE}.")
    w(f"- Total tokens per epoch (train, prefix included): 4B {sum(l4):,}; Lite {sum(ll):,}.")
    pre = {}
    for fam, key in (("4b", "pack"), ("lite", "pack_lite")):
        for r in allrows:
            pk = r["meta"].get(key)
            if pk is None or (fam == "lite" and "lite" not in r["meta"]["family_ok"]):
                continue
            nm = r["meta"]["pack_name"] if fam == "4b" else r["meta"]["pack_lite_name"]
            if (fam, nm) not in pre:
                sys_text = F.system_agent(r["meta"]["bio"]) if pk else F.SYSTEM_CHAT
                pre[(fam, nm)] = tok.prefix(sys_text, pk, "Face ID", fam)
    w("- System + tool block prefix (Face ID, no name/notes): " + ", ".join(f"{f} {n} {v:,}" for (f, n), v in sorted(pre.items())) + ".")
    top4 = sorted(train, key=lambda r: -r["meta"]["tokens_4b"])[:3]
    topl = sorted([r for r in train if "lite" in r["meta"]["family_ok"]], key=lambda r: -r["meta"]["tokens_lite"])[:3]
    w("- Longest 4B rows: " + "; ".join(f"{r['meta']['tokens_4b']:,} ({r['meta']['source']}/{r['meta'].get('kind')}, {r['meta']['pack_name']})" for r in top4) + ".")
    w("- Longest Lite rows: " + "; ".join(f"{r['meta']['tokens_lite']:,} ({r['meta']['source']}/{r['meta'].get('kind')}, {r['meta']['pack_lite_name']})" for r in topl) + ".\n")
    bios = Counter(r["meta"]["bio"] for r in train)
    w(f"Device word (train): " + ", ".join(f"{k} {v:,}" for k, v in bios.most_common()) + ".\n")
    w("Validation: `gen4/check4.py` (PLAN_v3 C9) runs over every row of both files; see its output for the result.\n")
    open(os.path.join(a.out_dir, "STATS.md"), "w").write("\n".join(L) + "\n")
    rep = {k: {str(kk): vv for kk, vv in v.items()} for k, v in st.items()}
    rep.update({"train": len(train), "val": len(val), "restraint_v2": v2_share, "restraint_sft5": share1, "topup": n_topup,
                "max_tokens_4b": max(l4), "max_tokens_lite": max(ll) if ll else None, "tokens_exact": tok.use})
    json.dump(rep, open(os.path.join(a.out_dir, "sft5_report.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
