#!/usr/bin/env python3
"""Fix the two known eval2/tools.jsonl bugs (NEXT_RUN.md section 6) in a COPY: eval3/tools_fixed.jsonl.

Cause (spec/NOTES_evalasm.md section 4): build_eval.py stored the generator's END-OF-RUN world snapshot, not the state
that the reference transcript saw. The reference transcript holds the right numbers in its own tool results, so the
world is rebuilt from them:
  * get_balances result  -> world.balances (rows the call returned replace the rows it filtered; no filter = all)
  * hyperliquid_account result -> world.hl_positions, hl_orders, hl_value (hl_value = account_value - sum(unrealised))
Affected kinds in the files: swap_all (12 of 14 items, not 4) and trade_chain_risk (all 14, not 4). The same cause also
dropped the failures that price_error, tool_error and tool_error_retry items rely on (sim.World.down / fail_next are reset
after use), so those are restored from the reference errors too. The script replays every
reference of every item (deterministic tools only) before and after, so any other kind with the same problem shows up in
the report. eval2/ is read only; this writes only eval3/tools_fixed.jsonl and eval3/tools_fixed_report.json.

    python gen3/fix_eval_tools.py
"""
import json, os, random, sys, copy
sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2 modules we import
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import sim                                                      # noqa: E402

SRC = os.path.join(LAB, "eval2", "tools.jsonl")
DST = os.path.join(LAB, "eval3", "tools_fixed.jsonl")
REP = os.path.join(LAB, "eval3", "tools_fixed_report.json")
DETERMINISTIC = ("get_balances", "hyperliquid_account", "get_price", "position_math", "calculator")


def ref_pairs(ref):
    """[(call, result_obj)] from a reference transcript (assistant tool_calls followed by tool messages)."""
    out = []
    for i, m in enumerate(ref):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for k, tc in enumerate(m["tool_calls"]):
                try:
                    res = json.loads(ref[i + 1 + k]["content"])
                except Exception:
                    res = None
                out.append((tc, res))
    return out


def mismatches(world_state, pairs):
    w = sim.World.from_state(world_state)
    bad = []
    for tc, res in pairs:
        if tc["name"] not in DETERMINISTIC or res is None:
            continue
        got = w.call(tc["name"], tc["arguments"])
        if got != res:
            bad.append(tc["name"])
    return bad


def fix_world(world_state, pairs):
    st = copy.deepcopy(world_state)
    for tc, res in pairs:
        if res is not None and "error" in res:
            # same cause: the snapshot lost the failure the reference saw (sim.World.down / fail_next are reset after use)
            err = str(res["error"])
            if err.startswith("price feed unavailable for "):
                st["down"] = sorted(set(st.get("down", [])) | {err.rsplit(" ", 1)[1]})
            elif err in ("network timeout, try again", "service unavailable", "request timed out") and not st.get("fail_next"):
                st["fail_next"] = tc["name"]
            continue
        if res is None:
            continue
        if tc["name"] == "get_balances":
            ch = (tc["arguments"] or {}).get("chain")
            keep = [b for b in st["balances"] if ch is not None and b["chain"] != str(ch).lower()]
            st["balances"] = keep + [dict(b) for b in res.get("balances", [])]
        elif tc["name"] == "hyperliquid_account":
            st["hl_positions"] = [dict(p) for p in res.get("positions", [])]
            st["hl_orders"] = [dict(o) for o in res.get("open_orders", [])]
            upnl = round(sum(p["unrealized_pnl"] for p in st["hl_positions"]), 2)
            st["hl_value"] = round(res["account_value"] - upnl, 2)
    return st


def main():
    rows = [json.loads(l) for l in open(SRC)]
    report = {"n": len(rows), "kinds": defaultdict(Counter), "still_bad": []}
    out = []
    for it in rows:
        pairs = ref_pairs(it["reference"])
        before = mismatches(it["world"], pairs)
        it2 = copy.deepcopy(it)
        if before:
            it2["world"] = fix_world(it["world"], pairs)
            after = mismatches(it2["world"], pairs)
            if it2["world"] != it["world"]:
                it2["fixed_world"] = True
        else:
            after = []
        k = report["kinds"][it["kind"]]
        k["items"] += 1
        k["bad_before"] += bool(before)
        k["bad_after"] += bool(after)
        if after:
            report["still_bad"].append({"id": it["id"], "tools": after})
        out.append(it2)
    report["kinds"] = {k: dict(v) for k, v in report["kinds"].items() if v["bad_before"] or v["bad_after"]}
    report["fixed_items"] = sum(v["bad_before"] - v["bad_after"] for v in report["kinds"].values())
    report["worlds_changed"] = sum(1 for it in out if it.get("fixed_world"))
    with open(DST, "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump(report, open(REP, "w"), indent=1)
    still = Counter(x["id"].rsplit("-", 1)[0] for x in report["still_bad"])
    print("worlds changed:", report["worlds_changed"], "| items whose reference now replays:", report["fixed_items"], "| kinds touched:", {k: (v["bad_before"], v["bad_after"]) for k, v in report["kinds"].items()})
    print("still different (not world bugs, see README): ", dict(still))


if __name__ == "__main__":
    main()
