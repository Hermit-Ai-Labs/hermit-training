#!/usr/bin/env python3
"""check4: PLAN_v3 C9 over every row of data5/sft5_train.jsonl and data5/sft5_val.jsonl. Exit 1 on any error.

Checks (name: rule; scope):
  schema        every call: known v3 tool, required args (prepare_send may omit chain), no unknown args, JSON types, enums,
                nested items; each tool_calls message is followed by exactly one tool message per call, each a JSON object.
  pack          meta.pack is a pack the app can send (router pack of either family, or all 30; <= 22 tools unless all 30);
                every call is in meta.pack; "lite" rows: pack_lite is a Lite router pack (<= 16 tools), holds every call,
                at most 3 calls, tokens_lite present. Chat rows (tools false) have pack [] and no calls.
  arg_omission  optional args only when the user said them: chain/network (prepare_send, prepare_multisend,
                schedule_payment, add_contact) named by a user turn (ChainWords) or an earlier result; receive / pay_with
                named in a user turn; cancel_order.kind matches the user's words (stop / take profit / order / limit) or an
                earlier needs_choice result; order_id from an earlier result; schedule_payment.repeat said (every, weekly,
                monthly, once ...); open_page.tab named by the user (or a synonym) or an earlier result; prepare_sell has
                percent XOR amount.
  mint          a mint-shaped prepare_buy.symbol / prepare_sell.token is in the user's typed text or an earlier
                search_tokens / get_positions result or needs_choice options.
  address       prepare_send.to_address and add_contact.address typed by the user (file text never counts) or returned by
                an earlier get_address {contact}; address-shaped schedule_payment.payee the same; multisend addresses typed
                or in an attached file.
  replay        calculator results equal gen2 calc_result(expression) byte for byte. (No sim4 exists: the full replay of
                every result against hermit_tools_v3.meta.json was run by each data4 generator's own self-check, and for
                sft4 rows by assemble4's conversions; this check is the part that can be re-run here.)
  tokens        tokens_4b <= 8192 and tokens_lite <= 6144 (recomputed with --tokens, else the values assemble5 measured).
  c6b           review results (review_opened / prepared_for_review): the answer says it is prepared / waiting / nothing
                moved; card results (card_opened): card wording and no "cancelled/saved/added/created/scheduled/set up/
                turned on/is now active" as fact; schedule_payment: "reminder", never automatic / will be sent / will pay /
                pays it for you / I'll send; search_tokens: never safe / legit / trustworthy / no risk / can't lose (unless
                negated), every refusal and, for a single-token result, every tag repeated; prepare_buy warnings repeated;
                open_page / open_add_funds: "button", never "I opened"; set_tp_sl / cancel_order: `replaces` / `warning`
                repeated, never "you're protected"; tools rows never "tab bar" or "go to X, then Y" / "go to X > Y".
  forbidden     PLAN C9.5 on assistant text: Relay, went through, has been sent/..., I've sent/..., successfully, funds are
                in/arrived, leading "Done", not financial advice, middle dots, asking the price / budget / "how much do
                you"; negated uses ("nothing has been sent", "I can't tell if it went through") and confirmed results
                (status sent / confirmed, executed true) are allowed. Plus lint5: never write it down, seed/key "not
                stored", impermanent loss + Aave, ".eth can't resolve", funding_8h, unfilled {BIO}.
  bio           meta.bio equals the system prompt's device word; the other word appears in no system, tool or assistant
                text (a user may type it).
  numbers       number audit (assemble4.number_audit: every $ amount, decimal and 4+ digit number in an assistant text
                comes from the system, a user turn, a file, an earlier result, call or calculator result; rounded copies
                count); rows with at least one tool call (knowledge answers quote facts, not results).
  words         generator rows (data4): answers <= 70 words, 110 for views / files / tables / lists, 250 when the user
                asks for detail. sft4 rows are exempt (their caps were checked by gen3).
  roundtrip     parse_calls(assistant_text) gives back the calls; render works for 4B (pack) and Lite (pack_lite).

    python gen4/check4.py                       # both files, stored token counts
    python gen4/check4.py --tokens              # recount tokens with the Qwen3 tokenizers
    python gen4/check4.py --show 3              # examples per failing check
"""
import argparse, json, os, re, sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import fmt4 as F                                   # noqa: E402
import packs as P                                  # noqa: E402
sys.path.insert(1, os.path.join(LAB, "gen3"))
import assemble4 as A4                             # noqa: E402  (number_audit, LINT)
import send_nochain as SN                          # noqa: E402  (CHAIN_WORDS, mentioned, find_chain, user_typed, without_files)

D5 = os.path.join(LAB, "data5")
B58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
EVM = re.compile(r"^0x[0-9a-fA-F]{40}$")
ADDR_ANY = re.compile(r"0x[0-9a-fA-F]{40}|(?<![1-9A-HJ-NP-Za-km-z])[1-9A-HJ-NP-Za-km-z]{32,44}(?![1-9A-HJ-NP-Za-km-z])")
REVIEW_STATUS = {"review_opened", "prepared_for_review"}
NEG = re.compile(r"\b(not|n't|no|nothing|never|until|unless|whether|if|before|can't|cannot|won't|isn't|aren't|wasn't|hasn't|haven't|don't|doesn't|didn't|nor|without)\b|n't\b", re.I)

# ------------------------------------------------------------------ forbidden (PLAN C9.5) and lint5
DONE_RX = [("went_through", re.compile(r"went through|gone through", re.I)),
           ("has_been", re.compile(r"\b(has|have) been (sent|swapped|bought|sold|deposited|withdrawn|closed|cancell?ed|saved|scheduled|created)\b", re.I)),
           ("i_did", re.compile(r"\bI(?:'ve| have)? (sent|bought|sold|swapped|deposited|withdrew|closed|cancell?ed)\b", re.I)),
           ("successfully", re.compile(r"\bsuccessfully\b", re.I)),
           ("funds_in", re.compile(r"\bfunds (are|have) (in|arrived)\b", re.I))]
OTHER_RX = [("relay", re.compile(r"\bRelay\b")),
            ("done_open", re.compile(r"^\s*done\b", re.I)),
            ("nfa", re.compile(r"not financial advice", re.I)),
            ("middle_dot", re.compile(r"[·•]")),
            ("never_write_down", A4.LINT[1][1]), ("not_stored", A4.LINT[2][1]), ("il_aave", A4.LINT[3][1]),
            ("eth_cant_resolve", A4.LINT[4][1]), ("funding_8h", re.compile(r"funding_8h")), ("unfilled_bio", re.compile(r"\{BIO\}")),
            ("tab_bar", re.compile(r"tab bar", re.I)), ("swap_in_wallets", re.compile(r"use Swap in Wallets", re.I))]
ASK_RX = [("ask_price", re.compile(r"what('s| is) (the )?(current )?price", re.I)), ("ask_budget", re.compile(r"\byour budget\b", re.I)),
          ("ask_how_much", re.compile(r"how much (do you|are you) (have|hold|own|holding|got)\b|what('s| is) your balance", re.I))]
NAV_RX = re.compile(r"\b(go to|head to|navigate to|open)\s+(the\s+)?(Trade|Wallets?|Earn|Settings|Payments|Explore|Home|Automations|Calendar|Projects|Files|Connectors)"
                    r"\s?>\s?\w+|\b(go to|head to|open)\s+(the\s+)?(Trade|Wallets?|Earn|Settings|Payments|Explore|Automations|Calendar|Projects|Files|Connectors)"
                    r"\b[^.\n]{0,30}?,?\s+(then|and then|and)\s+(tap|open|go to|select|choose|press|click|find|scroll)\b", re.I)

CARD_FACT = re.compile(r"\b(cancell?ed|saved|added|created|scheduled|set up|turned on|is now active)\b", re.I)
CARD_WORDING = re.compile(r"\bcard\b|\b(tap|click)(?:s|ping)? (on )?(Confirm|Create|Schedule|Save)\b|\bI'?ve drafted\b", re.I)
REVIEW_WORDING = re.compile(r"\bprepared?\b|waiting for (your|you)|nothing has (moved|been)|\breview\b|\bapprov", re.I)
SCHED_BAD = re.compile(r"\bautomatic(ally)?\b|will be sent|\bwill pay\b|pays it for you|\bI'll send\b", re.I)
TOKEN_BAD = re.compile(r"\b(safe|legit|trustworthy|no risk|can't lose)\b", re.I)
PROTECTED = re.compile(r"\byou'?re (now )?(fully )?protected\b|\byou are (now )?(fully )?protected\b", re.I)
OPENED = re.compile(r"\bI(?:'ve| have)? opened\b|\bopened (it|the page|the screen|that) for you\b", re.I)
TAG_KEYS = {"Freeze on": r"freez", "Mint on": r"\bmint", "Low liquidity": r"liquidity", "New": r"\bnew\b|recent|just launched|young", "Unverified": r"unverified|not verified",
            "Top holders": r"top holders|holders"}
WARN_KEYS = [(r"more than half of your balance", r"half"), (r"Unverified token", r"unverified|not verified"), (r"Mint authority", r"mint"),
             (r"New token", r"\bnew\b"), (r"price impact", r"impact"), (r"same address as row", r"same address|twice|duplicate"),
             (r"large amount", r"large"), (r"Top holders own", r"holders"), (r"Freeze authority", r"freez")]

TAB_SYN = {"positions": r"position|\bstop|take[- ]?profit|\btp\b|\bsl\b|order|close|protect|long|short|deposit|aave|earn|exit|trade|limit|liquidat",
           "markets": r"market|perp|stock|funding|list",
           "save": r"\bsav|deposit|aave|earn|lend|yield|apy",
           "explore": r"explor|pool|find|browse|opportunit|yield|option|apy|where|best|lend|deposit|put .{0,30} (in|into)|protocol|morpho",
           "schedule": r"schedul|upcoming|planned|recurring|remind|rmind|remnd|every|monthly|weekly|\bon (the )?\d|due|payment|tomorrow|next week|in \d+ days|(mon|tues|wednes|thurs|fri|satur|sun)day|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.? ?\d",
           "payees": r"payee|c\w{0,2}nt\w{0,2}ct|address book|saved|recipient|\bsave\b|add .* to my|people|\bsend\b|\bpay\b",
           "transfer": r"transfer|move|between|own wallet|other wallet|my wallets|each|batch|several|multiple|everyone|all of them", "apis": r"\bapi", "profile": r"profile|name",
           "safety": r"safety|limit|security|review|approval|face id|touch id|biometric|per day|daily|how much can", "memory": r"memory|note|remember|forget|hermit\.md|prefer|keep in mind|from now on|always",
           "models": r"model", "cloud": r"cloud"}
KIND_WORDS = {"stop_loss": r"\bstop|\bsl\b", "take_profit": r"take[- ]?profit|\btp\b|target|profit", "order": r"\border|\blimit"}
REPEAT_WORDS = {"monthly": r"month|monthly|\b1st\b|first of|every", "weekly": r"week|weekly|every (mon|tues|wednes|thurs|fri|satur|sun)day|every",
                "once": r"\bonce\b|one[- ]time|just this|only this|single"}


def dec(s):
    try:
        return Decimal(str(s).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def sentence_at(text, i):
    a = max(text.rfind(". ", 0, i), text.rfind("\n", 0, i), text.rfind("? ", 0, i), text.rfind("! ", 0, i))
    b = min([x for x in (text.find(". ", i), text.find("\n", i), text.find("? ", i), text.find("! ", i)) if x >= 0] or [len(text)])
    return text[a + 1:b + 1]


def negated_before(text, m, window=60):
    s = sentence_at(text, m.start())
    pre = s[:max(0, s.find(m.group(0)))] if m.group(0) in s else text[max(0, m.start() - window):m.start()]
    return bool(NEG.search(pre[-window:]))


def is_question(text, m):
    s = sentence_at(text, m.start())
    return s.rstrip().endswith("?") or (text.find("?", m.end()) != -1 and "." not in text[m.end():text.find("?", m.end())])


NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
             "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "half": 50, "quarter": 25, "third": 33.33,
             "dozen": 12, "grand": 1000}
SCALE = {"k": Decimal(10) ** 3, "m": Decimal(10) ** 6, "b": Decimal(10) ** 9, "t": Decimal(10) ** 12}
SRC_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?\s?([kKmMbBtT]\b)?")
STRIP = re.compile(r"```.*?```|`[^`\n]*`|https?://\S+|0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44}|0x[0-9a-fA-F]{2,8}(?:…|\.\.\.)[0-9a-fA-F]{2,8}"
                   r"|\b[1-9A-HJ-NP-Za-km-z]{3,8}(?:…|\.\.\.)[1-9A-HJ-NP-Za-km-z]{3,8}\b|\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?\b|\b\d{1,2}:\d{2}\b", re.S)


def number_audit(msgs):
    """assemble4.number_audit, extended: e-notation and $5K-style sources, number words in user text ("forty bucks"),
    decimal commas in non-English answers (751,37), addresses / shortened addresses / urls / dates / times / code skipped."""
    src = set()

    def add_num(d):
        if d is not None and d.is_finite() and abs(d) < Decimal("1e30"):
            src.add(d); src.add(abs(d))

    def add(text):
        for mm in SRC_NUM.finditer(text or ""):
            raw = mm.group(0).rstrip("kKmMbBtT ").strip()
            d = dec(raw)
            if d is None:
                try:
                    d = Decimal(repr(float(raw.replace(",", ""))))
                except Exception:
                    d = None
            try:
                if d is None or not d.is_finite() or d.adjusted() > 30 or d.adjusted() < -30:
                    continue
                add_num(d.normalize())
            except ArithmeticError:
                continue
            if mm.group(1):
                add_num((d * SCALE[mm.group(1).lower()]).normalize())
            if "," in raw and re.fullmatch(r"-?\d+,\d{1,4}", raw):           # decimal comma in the source
                add_num(dec(raw.replace(",", ".")))
        for w in re.findall(r"[a-z]+", (text or "").lower()):
            if w in NUM_WORDS:
                add_num(Decimal(str(NUM_WORDS[w])))

    for m in msgs:
        t = m.get("content") or ""
        if m["role"] == "assistant" and t:
            body = STRIP.sub(" ", t)
            for mm in re.finditer(r"(\$)?(?<![\w.,])(-?\d[\d,]*(?:\.\d+)?)\s?([kKmMbBtT]\b)?", body):
                s_ = mm.group(2).rstrip(",")
                if not (mm.group(1) or "." in s_ or len(s_.replace(",", "").lstrip("-")) >= 4 or re.fullmatch(r"-?\d+,\d{1,2}", s_)):
                    continue
                cands = []
                if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s_) or "," not in s_:
                    cands.append(s_.replace(",", ""))
                if re.fullmatch(r"-?\d+,\d+", s_):
                    cands.append(s_.replace(",", "."))                   # decimal comma
                ok = False
                for c in cands:
                    d = dec(c)
                    if d is None:
                        continue
                    k = len(c.split(".")[1]) if "." in c else 0
                    half = Decimal(5) * Decimal(10) ** (-k - 1)
                    sc = SCALE.get((mm.group(3) or "").lower())
                    if sc:
                        d, half = d * sc, half * sc
                    if d in src or abs(d) in src or any(abs(abs(x) - abs(d)) <= half for x in src if abs(x) < Decimal("1e15")):
                        ok = True
                        break
                if not ok:
                    return s_
        add(t)
        for tc in m.get("tool_calls") or []:
            add(json.dumps(tc["arguments"]))
    return None


def words(t):
    return len(re.findall(r"\S+", re.sub(r"```.*?```", " CODE ", t, flags=re.S)))


# ------------------------------------------------------------------ per-row walk
def walk(row):
    """Yield (kind, index, message, info). info for tool messages: (call, parsed result or None)."""
    queue, out = [], []
    for i, m in enumerate(row["messages"]):
        if m["role"] == "assistant" and m.get("tool_calls"):
            queue = list(m["tool_calls"])
            out.append(("calls", i, m, None))
        elif m["role"] == "tool":
            call = queue.pop(0) if queue else None
            try:
                o = json.loads(m["content"])
            except Exception:
                o = None
            out.append(("tool", i, m, (call, o)))
        else:
            out.append((m["role"], i, m, None))
    return out


GUARD_DEMO = [0]


def check_row(row, tok=None, generator_sources=None):
    errs = []
    E = lambda name, msg: errs.append((name, msg))
    msgs, meta = row["messages"], row["meta"]
    tools = row["tools"]
    src = meta.get("source")
    bio = meta.get("bio")
    steps = walk(row)
    calls = [tc for k, _, m, _ in steps if k == "calls" for tc in m["tool_calls"]]
    names = [c["name"] for c in calls]

    # ---------------- structure + schema
    if not msgs or msgs[0]["role"] != "system":
        E("schema", "no system message")
    for k, i, m, _ in steps:
        if k == "calls":
            n = len(m["tool_calls"])
            nxt = msgs[i + 1:i + 1 + n]
            if len(nxt) != n or any(x["role"] != "tool" for x in nxt):
                E("schema", f"message {i}: {n} calls not followed by {n} tool messages")
            for tc in m["tool_calls"]:
                for e in F.validate_call(tc):
                    E("schema", f"{tc['name']}: {e}")
        if k == "tool" and not isinstance(steps[[s[1] for s in steps].index(i)][3][1], dict):
            E("schema", f"message {i}: tool result is not a JSON object")
    if msgs and msgs[-1]["role"] != "assistant":
        E("schema", "row does not end with an assistant message")

    # ---------------- pack / family
    pack = meta.get("pack")
    fam = meta.get("family_ok")
    if pack is None or fam not in (["4b"], ["4b", "lite"]):
        E("pack", f"meta.pack / family_ok missing or odd: {fam}")
    else:
        if tools:
            if not (P.is_valid(pack, "4b") or P.is_valid(pack, "lite")):
                E("pack", f"meta.pack is not a pack the app sends ({len(pack)} tools)")
            if F.norm_pack(pack) != pack:
                E("pack", "meta.pack not in registry order")
            out = sorted(set(names) - set(pack))
            if out:
                E("pack", f"calls outside meta.pack: {out}")
        else:
            if pack != [] or names:
                E("pack", "chat row with tools or calls")
        if "lite" in fam:
            pl = meta.get("pack_lite")
            if pl is None:
                E("pack", "lite row without pack_lite")
            else:
                if tools and not P.is_valid(pl, "lite"):
                    E("pack", "pack_lite is not a Lite router pack")
                if set(names) - set(pl):
                    E("pack", f"calls outside pack_lite: {sorted(set(names) - set(pl))}")
                if len(pl) > 16:
                    E("pack", "pack_lite over 16 tools")
            if len(names) > 3:
                E("pack", "lite row with more than 3 calls")
            if not meta.get("tokens_lite"):
                E("pack", "lite row without tokens_lite")

    # ---------------- provenance walk (user text, results so far)
    refused_after = {}
    guard_demo = GUARD_DEMO
    for k, i, m, info in steps:
        if k == "tool" and info[0] is not None and isinstance(info[1], dict) and str(info[1].get("error", "")).startswith("Refused"):
            refused_after[id(info[0])] = True
    typed, alltext, results, prev_results_text = "", "", [], ""
    searched, contact_addrs, choice_text = set(), set(), ""
    for k, i, m, info in steps:
        if k == "user":
            typed += "\n" + SN.without_files(m["content"])
            alltext += "\n" + m["content"]
        elif k == "tool":
            call, o = info
            prev_results_text += "\n" + m["content"]
            if call and isinstance(o, dict):
                if call["name"] in ("search_tokens", "get_positions") or o.get("status") == "needs_choice" or o.get("options"):
                    for a in ADDR_ANY.findall(m["content"]):
                        searched.add(a)
                if call["name"] == "get_address" and call["arguments"].get("contact"):
                    for c in o.get("contacts") or []:
                        if c.get("address"):
                            contact_addrs.add(c["address"])
                if o.get("status") == "needs_choice" or o.get("question"):
                    choice_text += "\n" + m["content"]
        elif k == "calls":
            for tc in m["tool_calls"]:
                n, a = tc["name"], tc["arguments"] if isinstance(tc["arguments"], dict) else {}
                said = typed.lower()
                # chain / network
                for key in ("chain", "network"):
                    if key in a and n in ("prepare_send", "prepare_multisend", "schedule_payment", "add_contact"):
                        cid = SN.find_chain(a[key])
                        ok = (cid and SN.mentions(cid, typed)) or bool(re.search(r"\b" + re.escape(str(a[key]).lower()) + r"\b", said))
                        if not ok:
                            E("arg_omission", f"{n}.{key}={a[key]!r} not named by the user")
                if n == "prepare_sell":
                    if ("percent" in a) == ("amount" in a):
                        E("arg_omission", "prepare_sell needs exactly one of percent / amount")
                    if "receive" in a and not re.search(r"\b" + re.escape(str(a["receive"]).lower()) + r"\b", said):
                        E("arg_omission", f"prepare_sell.receive={a['receive']} not said")
                if n == "prepare_buy" and "pay_with" in a and not re.search(r"\b" + re.escape(str(a["pay_with"]).lower()) + r"\b", said):
                    E("arg_omission", f"prepare_buy.pay_with={a['pay_with']} not said")
                if n == "cancel_order":
                    if "kind" in a and not (re.search(KIND_WORDS.get(a["kind"], "$^"), said) or a["kind"] in choice_text):
                        E("arg_omission", f"cancel_order.kind={a['kind']} not said")
                    if "order_id" in a and str(a["order_id"]) not in prev_results_text:
                        E("arg_omission", f"cancel_order.order_id {a['order_id']} not from a result")
                if n == "schedule_payment" and "repeat" in a and not re.search(REPEAT_WORDS.get(a["repeat"], "$^"), said):
                    E("arg_omission", f"schedule_payment.repeat={a['repeat']} not said")
                if n == "open_page" and "tab" in a:
                    t = str(a["tab"]).lower()
                    if not (t in said or re.search(TAB_SYN.get(t, re.escape(t)), said) or t in prev_results_text.lower()):
                        E("arg_omission", f"open_page.tab={t} not asked for")
                # mint provenance
                for key, tool in (("symbol", "prepare_buy"), ("token", "prepare_sell")):
                    v = str(a.get(key, ""))
                    if n == tool and B58.match(v) and re.search(r"\d", v):
                        if not (SN.user_typed(v, typed) or v in searched):
                            E("mint", f"{tool}.{key} mint not typed and not from a search result")
                # address provenance
                refused = refused_after.get(id(tc), False)
                if refused:
                    guard_demo[0] += 1
                if n == "prepare_send" and not refused:
                    v = str(a.get("to_address", ""))
                    if not (SN.user_typed(v, typed) or v in contact_addrs):
                        E("address", "prepare_send.to_address neither typed nor from get_address {contact}")
                if n == "add_contact" and not refused:
                    v = str(a.get("address", ""))
                    if not SN.user_typed(v, typed):
                        E("address", "add_contact.address not typed by the user")
                if n == "schedule_payment" and not refused:
                    v = str(a.get("payee", ""))
                    if ADDR_ANY.fullmatch(v) and not (SN.user_typed(v, typed) or v in contact_addrs):
                        E("address", "schedule_payment.payee address neither typed nor from get_address")
                if n == "prepare_multisend":
                    for rcp in a.get("recipients") or []:
                        v = str(rcp.get("address", rcp.get("to_address", "")))
                        if v and not (SN.user_typed(v, typed) or v in alltext):
                            E("address", "multisend address neither typed nor in a file")
                # replay: calculator
                if n == "calculator":
                    pass

    for k, i, m, info in steps:
        if k == "tool" and info[0] and info[0]["name"] == "calculator":
            want = F._H.calc_result(info[0]["arguments"].get("expression", ""))
            if json.loads(want) != info[1]:
                E("replay", f"calculator {info[0]['arguments'].get('expression')!r}: recorded {m['content']} != {want}")

    # ---------------- tokens
    if tok is not None:
        t4 = tok.lengths(row, pack or [], "4b")[0]
        if t4 != meta.get("tokens_4b"):
            E("tokens", f"tokens_4b stored {meta.get('tokens_4b')} != recount {t4}")
        if fam and "lite" in fam:
            tl = tok.lengths(row, meta["pack_lite"], "lite")[0]
            if tl != meta.get("tokens_lite"):
                E("tokens", f"tokens_lite stored {meta.get('tokens_lite')} != recount {tl}")
    if (meta.get("tokens_4b") or 0) > 8192:
        E("tokens", f"4B {meta['tokens_4b']} > 8192")
    if fam and "lite" in fam and (meta.get("tokens_lite") or 0) > 6144:
        E("tokens", f"Lite {meta['tokens_lite']} > 6144")

    # ---------------- assistant text checks
    def is_confirmed(o):
        return isinstance(o, dict) and (o.get("executed") is True or o.get("status") in ("sent", "confirmed") or any(
            isinstance(r, dict) and r.get("status") == "confirmed" for r in (o.get("rows") or o.get("activity") or o.get("items") or [])))
    turn_results = []
    is_gen = generator_sources is not None and src in generator_sources
    last_user = ""
    for k, i, m, info in steps:
        if k == "user":
            turn_results = []
            last_user = m["content"]
            continue
        if k == "tool":
            turn_results.append(info)
            continue
        if k not in ("assistant", "calls"):
            continue
        t = m.get("content") or ""
        if not t:
            continue
        # forbidden + lint
        tcode = re.sub(r"```.*?```", "", t, flags=re.S)
        for name, rx in DONE_RX:
            for mm in rx.finditer(tcode):
                confirmed = any(is_confirmed(o) for _, o in turn_results)     # this turn's results only
                if confirmed or negated_before(tcode, mm) or re.search(r"\b(confirm|check|verify|see|whether|once|when|after|until|unless|in case|make sure|shows?)\b[^.]{0,30}$", tcode[max(0, mm.start() - 40):mm.start()], re.I):
                    continue
                t_ = tcode
                E("forbidden", f"{name}: ...{t_[max(0, mm.start() - 50):mm.end() + 20]!r}")
        money_turn = any(c and (c["name"].startswith("prepare_") or c["name"] in ("set_tp_sl", "cancel_order", "create_automation",
                                                                                      "schedule_payment", "add_contact"))
                         for c, _ in turn_results)
        for name, rx in OTHER_RX:
            tt = re.sub(r"```.*?```", "", t, flags=re.S)
            if name in ("tab_bar",) and not tools:
                continue
            if name == "done_open" and not money_turn:
                continue
            if name == "il_aave":
                mm = rx.search(tt)
                if mm and re.search(r"\b(no|not|n't|without|isn't|doesn't|never|zero|none)\b[^.]{0,40}impermanent loss|impermanent loss[^.]{0,30}\b(doesn't|does not|isn't|is not|n't) (apply|affect|happen)", tt, re.I):
                    continue
            mm = rx.search(tt)
            if mm:
                E("forbidden", f"{name}: ...{tt[max(0, mm.start() - 40):mm.end() + 20]!r}")
        for name, rx in ASK_RX:
            for mm in rx.finditer(t):
                if is_question(t, mm) and tools:
                    E("forbidden", f"{name}: ...{t[max(0, mm.start() - 40):mm.end() + 30]!r}")
        if tools:
            mm = NAV_RX.search(t)
            if mm:
                E("c6b", f"navigation instead of open_page: {mm.group(0)!r}")
        if k == "calls":
            continue
        # c6b on the answer after this turn's results
        for call, o in turn_results:
            if not call or not isinstance(o, dict):
                continue
            n, st = call["name"], o.get("status")
            english = sum(ch.isascii() for ch in t) / max(1, len(t)) > 0.97 and meta.get("lang", "en") in ("en", None)
            if st in REVIEW_STATUS and english and not REVIEW_WORDING.search(t):
                E("c6b", f"{n} review answer lacks 'prepared'/'waiting'/'nothing has moved': {t[:120]!r}")
            if st == "card_opened":
                if english and not CARD_WORDING.search(t):
                    E("c6b", f"{n} card answer lacks card wording: {t[:120]!r}")
                for mm in CARD_FACT.finditer(t):
                    after = t[mm.end():mm.end() + 30].lower()
                    before = t[max(0, mm.start() - 12):mm.start()].lower()
                    if negated_before(t, mm) or re.match(r"\s*(once|until|when|after|only when|as soon as|if|the moment|on your tap|by tapping)", after) \
                            or re.search(r"\b(to|be|get|gets|is|are)\s+$", before) and re.search(r"\b(to|be|get|gets)\s+$", before) \
                            or re.search(r"\b(want|wish|like)\b[^.]{0,30}$", before):
                        continue
                    E("c6b", f"{n} card answer states {mm.group(0)!r} as fact: ...{t[max(0, mm.start() - 50):mm.end() + 25]!r}")
                if PROTECTED.search(t):
                    E("c6b", "'you're protected' before approval")
            if n == "schedule_payment" and st == "card_opened":
                if english and "reminder" not in t.lower():
                    E("c6b", "schedule_payment answer without 'reminder'")
                mm = SCHED_BAD.search(t)
                if mm and not negated_before(t, mm):
                    E("c6b", f"schedule_payment answer says {mm.group(0)!r}")
            if n == "search_tokens" and isinstance(o.get("tokens"), list):
                for mm in TOKEN_BAD.finditer(t):
                    if not negated_before(t, mm, 80):
                        E("c6b", f"search_tokens answer calls a token {mm.group(0)!r}: ...{t[max(0, mm.start() - 40):mm.end() + 20]!r}")
                toks = o["tokens"]
                for tk in toks:
                    if tk.get("refusal") and not re.search(r"won'?t buy|will not buy|can'?t buy|isn'?t buying|not buy", t, re.I) \
                            and len(toks) == 1:
                        E("c6b", "search_tokens refusal not repeated")
                if len(toks) == 1:
                    for tag in toks[0].get("tags") or []:
                        if tag.startswith("Top holders") and toks[0].get("verified"):
                            continue
                        key = next((v for k2, v in TAG_KEYS.items() if tag.startswith(k2)), re.escape(tag))
                        if not re.search(key, t, re.I):
                            E("c6b", f"search_tokens tag {tag!r} not repeated")
            if isinstance(o.get("warnings"), list) and n in ("prepare_buy", "prepare_sell", "prepare_swap", "prepare_send", "prepare_multisend"):
                for wtxt in o["warnings"]:
                    key = next((v for k2, v in WARN_KEYS if re.search(k2, wtxt, re.I)), None)
                    if key and not re.search(key, t, re.I):
                        E("c6b", f"{n} warning not repeated: {wtxt!r}")
            if n in ("open_page", "open_add_funds") and st == "button_shown":
                if english and "button" not in t.lower():
                    E("c6b", f"{n} answer without 'button': {t[:100]!r}")
                if OPENED.search(t):
                    E("c6b", f"{n} answer says it opened the page")
            if n in ("set_tp_sl", "cancel_order") and st in ("review_opened", "card_opened"):
                if o.get("replaces"):
                    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", o["replaces"])
                    if nums and not any(x in t for x in nums):
                        E("c6b", f"{n}: replaces {o['replaces']!r} not mentioned")
                if o.get("warning") and not re.search(r"without it|no stop|no take profit|unprotected|no protection|nothing (to )?protect|not protected|lose (the|its|your) (stop|protection)", t, re.I):
                    E("c6b", f"{n}: warning {o['warning']!r} not repeated")
                if PROTECTED.search(t):
                    E("c6b", "'you're protected' before approval")
        # words (generator rows)
        if is_gen and m["role"] == "assistant":
            lu = last_user.lower()
            if re.search(r"detail|explain|in depth|elaborate|walk me|step by step|full|break ?down|everything|more|why|how does|how do|what happens|compare|difference|table|list|all of|each", lu):
                cap = 250
            elif "<file>" in last_user or "|" in t or re.search(r"view|multi|table|list|summary|brief_long|detail", str(meta.get("kind", ""))) or re.search(r"\b(view|take|thoughts|should i|worth)\b", lu) or t.count("\n") >= 2:
                cap = 110
            else:
                cap = 70
            if words(t) > cap:
                E("words", f"{words(t)} words > {cap}")

    # ---------------- bio
    if tools or msgs[0]["content"].startswith(F.system_agent("Face ID")[:60]):
        sysw = F.detect_bio(msgs)
        if bio != sysw:
            E("bio", f"meta.bio {bio} != system {sysw}")
        other = F.other_bio(bio or "Face ID")
        for m in msgs:
            if m["role"] in ("system", "tool", "assistant") and other in (m.get("content") or ""):
                E("bio", f"{other} in a {m['role']} message")
                break

    # ---------------- numbers
    if names:
        bad = number_audit(msgs)
        if bad:
            E("numbers", f"answer number {bad} has no source")

    # ---------------- roundtrip + render
    for k, i, m, _ in steps:
        if k == "calls":
            got, badp = F.parse_calls(F.assistant_text(m))
            if badp or got != [{"name": c["name"], "arguments": c["arguments"]} for c in m["tool_calls"]]:
                E("roundtrip", f"message {i}: calls do not survive render/parse")
    try:
        F.render(msgs, pack or [], "4b", bio=bio)
        if fam and "lite" in fam:
            F.render(msgs, meta["pack_lite"], "lite", bio=bio)
    except Exception as e:
        E("roundtrip", f"render failed: {e}")
    return errs


def selftest(path, n=400):
    """Mutation check: broken copies of real rows must fail. Returns {mutation: (caught, tried)}."""
    import copy, random
    rng = random.Random(4)
    rows = [json.loads(l) for l in open(path)]
    rng.shuffle(rows)
    gen = None
    import assemble5 as AS
    gen = set(AS.DATA4 + AS.EXTRA4)

    def last_answer(r):
        for m in reversed(r["messages"]):
            if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
                return m
        return None

    def has(r, pred):
        return any(pred(tc) for m in r["messages"] for tc in (m.get("tool_calls") or []))

    def last_turn(r):
        idx = max(i for i, m in enumerate(r["messages"]) if m["role"] == "user")
        return r["messages"][idx + 1:]

    def last_has(r, pred):
        return any(pred(tc) for m in last_turn(r) for tc in (m.get("tool_calls") or []))

    def last_tool_has(r, text):
        return any(m["role"] == "tool" and text in m["content"] for m in last_turn(r))

    users = lambda r: " ".join(m["content"].lower() for m in r["messages"] if m["role"] == "user")

    muts = {
        "false_done": (lambda r: last_has(r, lambda tc: tc["name"].startswith("prepare_")) and not last_tool_has(r, '"confirmed"')
                       and not last_tool_has(r, '"executed": true'),
                       lambda r: last_answer(r).__setitem__("content", "I've sent it. " + last_answer(r)["content"])),
        "relay": (lambda r: has(r, lambda tc: tc["name"] in ("prepare_swap", "get_quote")),
                  lambda r: last_answer(r).__setitem__("content", last_answer(r)["content"] + " Routed via Relay.")),
        "invented_number": (lambda r: has(r, lambda tc: tc["name"] in ("get_price", "get_balances")),
                            lambda r: last_answer(r).__setitem__("content", last_answer(r)["content"] + " That is $91,234.57 in total.")),
        "chain_added": (lambda r: has(r, lambda tc: tc["name"] == "prepare_send" and "chain" not in tc["arguments"])
                        and not SN.mentions("polygon", users(r)),
                        lambda r: [tc["arguments"].__setitem__("chain", "polygon") for m in r["messages"] for tc in (m.get("tool_calls") or [])
                                   if tc["name"] == "prepare_send"]),
        "other_bio": (lambda r: r["tools"], lambda r: last_answer(r).__setitem__("content", last_answer(r)["content"] + " Approve with " + F.other_bio(r["meta"]["bio"]) + ".")),
        "tool_outside_pack": (lambda r: r["tools"] and "set_tp_sl" not in r["meta"]["pack"] and has(r, lambda tc: True),
                              lambda r: next(m for m in r["messages"] if m.get("tool_calls"))["tool_calls"].__setitem__(0, {"name": "set_tp_sl", "arguments": {"coin": "SOL", "stop_loss": 1}})),
        "card_done": (lambda r: any(m["role"] == "tool" and '"card_opened"' in m["content"] for m in r["messages"]),
                      lambda r: last_answer(r).__setitem__("content", "It's saved and scheduled. " + last_answer(r)["content"])),
        "opened": (lambda r: last_tool_has(r, '"button_shown"'),
                   lambda r: last_answer(r).__setitem__("content", "I opened the page for you.")),
        "bad_enum": (lambda r: has(r, lambda tc: tc["name"] == "open_page"),
                     lambda r: [tc["arguments"].__setitem__("page", "seedphrase") for m in r["messages"] for tc in (m.get("tool_calls") or []) if tc["name"] == "open_page"]),
        "untyped_address": (lambda r: has(r, lambda tc: tc["name"] == "prepare_send") and not any(m["role"] == "tool" and "Refused" in m["content"] for m in r["messages"]),
                            lambda r: [tc["arguments"].__setitem__("to_address", "0x" + "ab" * 20) for m in r["messages"] for tc in (m.get("tool_calls") or []) if tc["name"] == "prepare_send"]),
        "middle_dot": (lambda r: last_answer(r) is not None, lambda r: last_answer(r).__setitem__("content", last_answer(r)["content"] + " · ok")),
        "calc_tamper": (lambda r: any(m["role"] == "tool" and '"result"' in m["content"] for m in r["messages"]),
                        lambda r: next(m for m in r["messages"] if m["role"] == "tool" and '"result"' in m["content"]).__setitem__("content", json.dumps({"result": "123456.789"}))),
    }
    out = {}
    for name, (pred, mut) in muts.items():
        tried = caught = 0
        for r in rows:
            if tried >= n:
                break
            try:
                if not pred(r) or last_answer(r) is None:
                    continue
            except Exception:
                continue
            if check_row(r, None, gen):
                continue                     # only clean rows are mutated
            r2 = copy.deepcopy(r)
            mut(r2)
            tried += 1
            caught += bool(check_row(r2, None, gen))
        out[name] = (caught, tried)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="mutation check of the checker on real rows")
    ap.add_argument("files", nargs="*", default=[os.path.join(D5, "sft5_train.jsonl"), os.path.join(D5, "sft5_val.jsonl")])
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    if a.selftest:
        res = selftest(a.files[0])
        for k, (c, t) in res.items():
            print(f"  mutation {k:18} caught {c}/{t}")
        miss = sum(t - c for c, t in res.values())
        print("selftest OK" if miss == 0 else f"selftest: {miss} mutated rows not caught")
        return 0 if miss == 0 else 1
    tok = None
    if a.tokens:
        import assemble5
        tok = assemble5.Tok(True)
    gen_sources = set(json.load(open(os.path.join(HERE, "..", "data5", "sft5_report.json"))).get("generator_sources", [])) \
        if os.path.exists(os.path.join(D5, "sft5_report.json")) else set()
    import assemble5 as AS
    gen_sources |= set(AS.DATA4 + AS.EXTRA4)
    total, bad_rows = 0, 0
    by_check, by_src = Counter(), defaultdict(Counter)
    examples = defaultdict(list)
    for path in a.files:
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                row = json.loads(line)
                total += 1
                errs = check_row(row, tok, gen_sources)
                if errs:
                    bad_rows += 1
                    for name, msg in errs:
                        by_check[name] += 1
                        by_src[name][f"{row['meta'].get('source')}/{row['meta'].get('kind')}"] += 1
                        if len(examples[name]) < a.show:
                            examples[name].append(f"{os.path.basename(path)}:{ln} [{row['meta'].get('origin')}] {msg}")
    print(f"check4: {total} rows, {bad_rows} with errors, {sum(by_check.values())} errors "
          f"(guard demonstrations allowed: {GUARD_DEMO[0]} calls whose result is the app's 'Refused' guard error)")
    for name, n in by_check.most_common():
        print(f"  {name}: {n}  ({', '.join(f'{k} {v}' for k, v in by_src[name].most_common(6))})")
        for e in examples[name]:
            print("      " + e[:300])
    if bad_rows:
        print("FAIL")
        return 1
    print("check4 OK: every row passes (schema, pack, arg_omission, mint, address, replay, tokens, c6b, forbidden, bio, numbers, words, roundtrip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
