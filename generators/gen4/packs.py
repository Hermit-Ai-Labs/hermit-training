#!/usr/bin/env python3
"""Tool packs and the router of spec/APP_TOOLS_v3.md section 4, as code. The lab data, the eval and the app (ToolPacks.choose)
must use this exact rule; spec/render_golden_v3.json carries `router_cases` (message -> pack) as the app's parity fixtures.

Packs (registry order inside each rendered block, see fmt4.tools_block):
  core (12)  calculator, get_price, get_balances, get_portfolio, get_address, get_activity, prepare_send, prepare_swap,
             get_quote, open_page, open_app, open_add_funds
  trade (+8) hyperliquid_account, position_math, search_markets, prepare_trade, prepare_close, cancel_order, set_tp_sl,
             get_positions                         (Lite trade pack: hyperliquid_account, prepare_trade, prepare_close, cancel_order)
  solana(+4) search_tokens, prepare_buy, prepare_sell, get_positions
  earn (+4)  get_yields, prepare_deposit, prepare_withdraw, get_positions
  pay (+4)   prepare_multisend, add_contact, schedule_payment, create_automation

choose(first_message, has_attachment=False, family="4b") -> list of tool names (registry order)
  1. Keywords (whole words, case-insensitive) on the first user message (the typed text; an attached file's body is not
     scanned). has_attachment: True for a csv/xlsx/xls/tsv attachment, or the attachment's file name (a spreadsheet name
     counts, any other file does not) -> pay.
  2. Each matching pack gets a score = number of keyword hits; ties break by the earliest hit in the message, then by the
     fixed order trade, solana, earn, pay.
  3. family "4b": core + the best packs, at most two, adding a pack only while the union stays at or under 22 tools
     (trade is 20 with core, so trade never pairs with another pack). family "lite": core + the single best pack, trade
     replaced by the Lite trade pack (16 tools at most). family "cloud": all 30. No match: core only.
The pack is chosen once, at the first message, and kept for the whole chat (KV-cache prefix). A tool the model needs but the
pack lacks: the model says it isn't available in this chat and calls open_page. Data rows whose calls fall outside the
chosen pack are rendered with smallest_containing(calls, family) instead (what the app holds after the user reopens the
chat from the open_page button).

Keyword lists are section 4's words plus their plain inflections (plural, -ing, -ed); additions beyond section 4 are marked
EXT and listed in EXTENSIONS so the app owner can copy or drop them together with this file.
"""
import json, os, re, sys
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
_TOOLS = json.load(open(os.path.join(LAB, "spec", "hermit_tools_v3.json")))
ORDER = [t["function"]["name"] for t in _TOOLS]

CORE = ["calculator", "get_price", "get_balances", "get_portfolio", "get_address", "get_activity", "prepare_send",
        "prepare_swap", "get_quote", "open_page", "open_app", "open_add_funds"]
PACK_TOOLS = {
    "trade": ["hyperliquid_account", "position_math", "search_markets", "prepare_trade", "prepare_close", "cancel_order",
              "set_tp_sl", "get_positions"],
    "solana": ["search_tokens", "prepare_buy", "prepare_sell", "get_positions"],
    "earn": ["get_yields", "prepare_deposit", "prepare_withdraw", "get_positions"],
    "pay": ["prepare_multisend", "add_contact", "schedule_payment", "create_automation"],
}
LITE_TRADE = ["hyperliquid_account", "prepare_trade", "prepare_close", "cancel_order"]
PACK_ORDER = ["trade", "solana", "earn", "pay"]
CAP = {"4b": 22, "lite": 16, "cloud": 30}
MAX_PACKS = {"4b": 2, "lite": 1, "cloud": 4}
assert sorted(set(CORE).union(*PACK_TOOLS.values())) == sorted(ORDER)

STOCK_WORDS = ["nvda", "nvidia", "tsla", "tesla", "aapl", "apple", "msft", "microsoft", "meta", "facebook", "amzn", "amazon",
               "googl", "goog", "google", "alphabet", "coinbase", "mstr", "microstrategy", "hood", "robinhood", "spx", "s&p",
               "s&p 500", "s&p500", "sp500", "nasdaq", "stock", "stocks", "shares"]          # stock/stocks/shares: EXT
KEYWORDS = {
    "trade": ["long", "longs", "longing", "short", "shorts", "shorting", "leverage", "leveraged", "perp", "perps",
              "perpetual", "perpetuals", "position", "positions", "stop", "stops", "stop loss", "stop-loss", "stoploss",
              "take profit", "take-profit", "tp", "sl", "tp/sl", "cancel order", "cancel my order", "limit order",
              "limit orders", "liquidation", "liquidated", "liq", "funding", "hyperliquid", "hl"] + STOCK_WORDS,
    "solana": ["buy", "buying", "sell", "selling", "sold", "token", "tokens", "mint", "memecoin", "memecoins", "meme coin",
               "pump", "pumpfun", "pump.fun", "jupiter", "safe"],
    "earn": ["aave", "deposit", "deposits", "depositing", "withdraw", "withdrawal", "withdrawing", "yield", "yields", "apy",
             "apr", "earn", "earning", "save", "savings", "lend", "lending"],                       # apr, lend, lending: EXT
    "pay": ["contact", "contacts", "payee", "payees", "schedule", "scheduled", "scheduling", "remind", "reminder",
            "reminders", "every day", "every week", "every month", "automation", "automations", "automate", "recurring",
            "every morning", "every evening", "every night", "every hour", "every monday", "every tuesday",
            "every wednesday", "every thursday", "every friday", "every saturday", "every sunday", "every weekday",
            "weekdays", "daily", "weekly", "monthly", "address book"],
}
# EXT: words beyond the section 4 list (section 4 has "every day/week/month"; these are the same intent)
EXTENSIONS = {"trade": ["stock", "stocks", "shares", "liq", "hl"], "earn": ["apr", "lend", "lending"],
              "pay": ["every morning", "every evening", "every night", "every hour", "every <weekday>", "every weekday",
                      "weekdays", "daily", "weekly", "monthly", "every N minutes/hours/days", "address book"]}
EVERY_N = re.compile(r"\bevery\s+\d+\s*(?:min|mins|minutes?|h|hrs?|hours?|days?|weeks?)\b", re.I)       # EXT (pay)
B58_MINT = re.compile(r"(?<![1-9A-HJ-NP-Za-km-z])[1-9A-HJ-NP-Za-km-z]{32,44}(?![1-9A-HJ-NP-Za-km-z])")
EVM_ADDR = re.compile(r"0x[0-9a-fA-F]{40}")
SHEET_EXT = (".csv", ".xlsx", ".xls", ".tsv")


def _kw_regex(words):
    alts = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])", re.I)


KW_RE = {p: _kw_regex(ws) for p, ws in KEYWORDS.items()}


def ordered(names):
    s = set(names)
    return [n for n in ORDER if n in s]


def pack_tools(packs, family="4b"):
    names = list(CORE)
    for p in packs:
        names += LITE_TRADE if (family == "lite" and p == "trade") else PACK_TOOLS[p]
    return ordered(names)


def _is_mint(tok):
    # a Solana mint/address: base58, 32-44 chars, with at least one digit and one letter (so plain long words never count)
    return bool(re.search(r"\d", tok)) and bool(re.search(r"[A-Za-z]", tok))


def hits(first_message, has_attachment=False):
    """pack -> (count, first position). Only packs with at least one hit."""
    text = first_message or ""
    no_evm = EVM_ADDR.sub(lambda m: " " * len(m.group(0)), text)        # an EVM address is never scanned as a base58 mint
    out = {}
    for p in PACK_ORDER:
        ms = [m.start() for m in KW_RE[p].finditer(text)]
        if p == "pay":
            ms += [m.start() for m in EVERY_N.finditer(text)]
            evm = [m.start() for m in EVM_ADDR.finditer(text)]
            sol = [m.start() for m in B58_MINT.finditer(no_evm) if _is_mint(m.group(0))]
            if len(set(EVM_ADDR.findall(text))) + len({m for m in B58_MINT.findall(no_evm) if _is_mint(m)}) >= 2:
                ms += (evm + sol)[:1]                                   # "many addresses": two or more distinct
            if has_attachment is True or (isinstance(has_attachment, str) and has_attachment.lower().endswith(SHEET_EXT)):
                ms.append(len(text))                                    # attachment counts as one hit, after the text
        if p == "solana":
            ms += [m.start() for m in B58_MINT.finditer(no_evm) if _is_mint(m.group(0))]
        if ms:
            out[p] = (len(ms), min(ms))
    return out


def choose_packs(first_message, has_attachment=False, family="4b"):
    """-> list of pack names (no core), best first."""
    if family == "cloud":
        return list(PACK_ORDER)
    h = hits(first_message, has_attachment)
    ranked = sorted(h, key=lambda p: (-h[p][0], h[p][1], PACK_ORDER.index(p)))
    chosen = []
    for p in ranked:
        if len(chosen) >= MAX_PACKS[family]:
            break
        if len(pack_tools(chosen + [p], family)) <= CAP[family]:
            chosen.append(p)
    return chosen


def choose(first_message, has_attachment=False, family="4b"):
    """The tool names the app sends for this chat (registry order)."""
    if family == "cloud":
        return list(ORDER)
    return pack_tools(choose_packs(first_message, has_attachment, family), family)


# ---------------------------------------------------------------- every pack the router can produce
def router_packs(family):
    """name -> tool list for every pack combination the router can return for `family` (cloud = all 30)."""
    if family == "cloud":
        return {"all": list(ORDER)}
    out = {"core": pack_tools([], family)}
    for p in PACK_ORDER:
        out["core+" + p] = pack_tools([p], family)
    if MAX_PACKS[family] >= 2:
        for i, a in enumerate(PACK_ORDER):
            for b in PACK_ORDER[i + 1:]:
                t = pack_tools([a, b], family)
                if len(t) <= CAP[family]:
                    out[f"core+{a}+{b}"] = t
    return out


VALID = {f: router_packs(f) for f in ("4b", "lite")}
ALL30 = list(ORDER)


def name_of(pack, family="4b"):
    """Name of a pack (tool list) among the router's packs for `family`, 'all' for the 30, else None."""
    s = set(pack or [])
    if s == set(ORDER):
        return "all"
    for k, v in VALID.get(family, {}).items():
        if set(v) == s:
            return k
    return None


def label(pack):
    """Display name of any pack: a 4B router name, 'core+trade_lite' for the Lite trade pack, 'all', 'chat' or 'custom'."""
    if not pack:
        return "chat"
    n = name_of(pack, "4b")
    if n:
        return n
    n = name_of(pack, "lite")
    if n:
        return n.replace("core+trade", "core+trade_lite")
    return "custom"


def is_valid(pack, family="4b"):
    """A pack the app can send: one the router returns, or all 30 (4B and cloud only: Private Cloud / OpenRouter)."""
    n = name_of(pack, family)
    if n == "all":
        return family != "lite"
    return n is not None


def smallest_containing(tools, family="4b", prefer=None):
    """Smallest router pack (by tool count, then the router's preference `prefer`, then pack order) that holds every tool in
    `tools`. 4B falls back to all 30; Lite returns None when no 16-tool pack holds them."""
    need = set(tools)
    cands = [(len(v), 0 if (prefer is not None and set(v) == set(prefer)) else 1, i, k, v)
             for i, (k, v) in enumerate(VALID[family].items()) if need <= set(v)]
    if cands:
        return sorted(cands)[0][4]
    if family == "lite":
        return None
    return list(ORDER)


ROUTER_CASES = [
    ("what's the price of sol", False), ("long btc 5x with $100", False), ("sell half my wif", False),
    ("deposit 50 usdc into aave on base", False), ("remind me to pay alice 50 usdc on friday", False),
    ("send 20 usdc to 0x52908400098527886E0F7030069857D2E4169EE7", False),
    ("buy $10 of EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", False), ("pay everyone in this sheet", "payroll.xlsx"),
    ("summarise this pdf", "notes.pdf"), ("set a stop loss on my sol long and sell my wif", False),
    ("withdraw my aave usdc and buy wif with it", False), ("every morning at 8 tell me the eth price", False),
    ("save alice as a contact and deposit 20 usdc to aave", False), ("what is a liquidation price", False),
    ("cancel my eth limit order", False), ("hi", False), ("how much is my portfolio worth", False),
    ("is this token safe 7Gk2xQ9dRvTnWb4mLpYc8sHfJ3uZaE5oNiB1KtVwXyDe", False), ("long nvda $50 3x", False),
    ("pay these: 0x52908400098527886E0F7030069857D2E4169EE7 10, 0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B 5", False),
]


def router_case_table():
    out = []
    for msg, att in ROUTER_CASES:
        row = {"first_message": msg, "has_attachment": att}
        for fam in ("4b", "lite", "cloud"):
            ps = choose_packs(msg, att, fam)
            row[fam] = {"packs": ps, "name": name_of(choose(msg, att, fam), fam) or "?", "n_tools": len(choose(msg, att, fam))}
        out.append(row)
    return out


if __name__ == "__main__":
    for f in ("4b", "lite"):
        print(f, {k: len(v) for k, v in VALID[f].items()})
    for r in router_case_table():
        print(f"{r['first_message'][:60]:62} 4b={r['4b']['name']:<18} lite={r['lite']['name']}")
