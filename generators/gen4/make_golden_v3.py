#!/usr/bin/env python3
"""Write spec/render_golden_v3.json: the app's parity fixtures for the v3 prompt.

For both families (4b, lite) and both device words, the exact rendered system turn (SYSTEM_AGENT_v3 + "\\n\\n" + tool block,
wrapped in <|im_start|>system ... <|im_end|>\\n) for: the full 30-tool block, core alone, and every pack combination the
router (gen4/packs.py) can return for that family. Each entry has the text, its sha256, character count and Qwen3 token
count. Also: the assistant prefix per family (Lite adds the empty think block), a short conversation rendered end to end per
family, the router fixture table (first message -> pack per family), and the v2 compatibility hash (fmt4 with the 19 v2
tools == gen3/fmt3).

    python gen4/make_golden_v3.py
"""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import fmt4 as F          # noqa: E402
import packs as P         # noqa: E402

OUT = os.path.join(LAB, "spec", "render_golden_v3.json")


def main():
    from transformers import AutoTokenizer
    tk = {"4b": AutoTokenizer.from_pretrained(os.path.join(LAB, "..", "base", "Qwen3-4B-Instruct-2507")),
          "lite": AutoTokenizer.from_pretrained(os.path.join(LAB, "..", "base", "Qwen3-1.7B"))}
    ntok = lambda s, fam: len(tk[fam](s, add_special_tokens=False)["input_ids"])
    entries = []
    for fam in ("4b", "lite"):
        packs = dict(P.VALID[fam])
        packs["all"] = list(P.ORDER)
        for bio in F.BIOS:
            for name, tools in packs.items():
                label = P.label(tools)
                text = F.system_prefix(F.system_agent(bio), tools, bio)
                entries.append({"family": fam, "bio": bio, "pack": label, "router_can_send": name != "all" or fam == "4b",
                                "n_tools": len(tools), "tools": tools, "sha256": F.sha(text), "chars": len(text),
                                "tokens": ntok(text, fam), "text": text})
    conv = [{"role": "system", "content": F.system_agent("Face ID")},
            {"role": "user", "content": "sell half my wif"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "prepare_sell", "arguments": {"token": "WIF", "percent": 50}}]},
            {"role": "tool", "content": json.dumps({"executed": False, "kind": "sell", "note": "Nothing has moved. The Sell review is open with this Jupiter quote; the user approves it with Face ID or cancels.", "price_impact_pct": 0.04, "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5, "status": "review_opened", "symbol": "WIF", "you_get": "about 5.48 USDC on Solana", "you_sell": "3.01 WIF (50% of your WIF)"}, ensure_ascii=False)},
            {"role": "assistant", "content": "I've prepared selling 3.01 WIF (50% of your WIF) for about 5.48 USDC on Solana through Jupiter. Price impact is 0.04%. Approve with Face ID or cancel. Nothing has been sold yet."},
            {"role": "user", "content": "thanks"}]
    examples = []
    for fam in ("4b", "lite"):
        pk = P.choose("sell half my wif", False, fam)
        txt = F.render_text(conv, pk, fam, add_generation_prompt=True, bio="Face ID")
        examples.append({"family": fam, "pack": P.label(pk), "messages": conv, "text": txt, "sha256": F.sha(txt), "tokens": ntok(txt, fam)})
    v2 = F.tools_block(F.V2_NAMES, "Face ID")
    doc = {
        "about": "Hermit v3 prompt goldens (app parity fixtures). The app must produce `text` byte for byte for the same family, device word "
                 "and pack. Built by gen4/make_golden_v3.py from spec/SYSTEM_AGENT_v3.txt and spec/hermit_tools_v3.json with gen4/fmt4.py; "
                 "packs and router: gen4/packs.py (APP_TOOLS_v3 section 4). The system turn does not depend on the family; Lite differs in "
                 "the assistant prefix (empty think block) and in which packs the router can send.",
        "registry_sha256": F.sha(open(F.TOOLS_PATH).read()),
        "system_template_sha256": F.sha(F.SYSTEM_TEMPLATE),
        "tool_order": list(P.ORDER),
        "packs": {"core": P.CORE, **P.PACK_TOOLS, "trade_lite": P.LITE_TRADE},
        "caps": {"4b": {"max_tools": P.CAP["4b"], "max_packs": P.MAX_PACKS["4b"]}, "lite": {"max_tools": P.CAP["lite"], "max_packs": 1}, "cloud": "all 30"},
        "assistant_prefix": {"4b": "<|im_start|>assistant\n", "lite": "<|im_start|>assistant\n" + F.THINK_EMPTY},
        "tool_response_format": "<|im_start|>user\\n<tool_response>\\n{json}\\n</tool_response>[\\n<tool_response>...]<|im_end|>\\n (consecutive tool results share one user turn)",
        "v2_compat": {"rule": "fmt4.tools_block(19 v2 names) == gen3/fmt3.tools_block() (checked by fmt4._selftest on sft4 rows)",
                      "sha256_face_id": F.sha(v2)},
        "router": {"keywords": P.KEYWORDS, "extensions_beyond_section4": P.EXTENSIONS,
                   "rule": "whole-word keyword hits on the first message (+ csv/xlsx attachment, 2+ addresses -> pay, base58 mint -> solana); "
                           "rank by hit count, then earliest hit, then trade/solana/earn/pay; 4B adds up to 2 packs while <= 22 tools; "
                           "Lite 1 pack (trade -> Lite trade pack); cloud all 30; no hit -> core",
                   "cases": P.router_case_table()},
        "entries": entries,
        "examples": examples,
    }
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}: {len(entries)} system turns (4b {sum(e['family'] == '4b' for e in entries)}, lite {sum(e['family'] == 'lite' for e in entries)}), "
          f"{os.path.getsize(OUT):,} bytes")
    for e in entries:
        if e["bio"] == "Face ID":
            print(f"  {e['family']:4} {e['pack']:18} {e['n_tools']:2} tools {e['chars']:6} chars {e['tokens']:5} tokens")


if __name__ == "__main__":
    main()
