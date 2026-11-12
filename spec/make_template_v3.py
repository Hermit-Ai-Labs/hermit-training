#!/usr/bin/env python3
"""Write spec/chat_template_4b_v3.jinja and spec/chat_template_lite_v3.jinja (Hermit 1.3 / 1.3 Lite, tools v3) and check them.

Start: export/chat_template_4b.jinja / export/chat_template_lite.jinja (read only; Lite has the empty think block in every
assistant prefix and in the generation prompt). The fixed 12-tool block is replaced by the v3 setup the app sends
(spec/APP_TOOLS_v3.md section 4, gen4/fmt4.py):
  * all 30 tools of spec/hermit_tools_v3.json are in the template, one JSON line each (fmt4.TOOL_LINE), in registry order;
  * `tools` passed as a list of tool dicts ({"type": "function", "function": {"name": ...}} or {"name": ...}) or of names
    renders only those tools, in registry order (= the pack, fmt4.tools_block(pack)); `tools` without names (e.g. [1] or
    True) renders all 30 (the cloud / "all" pack);
  * the device word: "Face ID" in the tool lines becomes "Touch ID" when the system message says Touch ID and not Face ID
    (fmt3.detect_bio, the app's rule of rewriting the whole prompt);
  * the system prompt (spec/SYSTEM_AGENT_v3.txt with name / hermit.md suffixes) is the first system message, as before.
Checks (exit 1 on any mismatch): for both families, template == fmt4.render_text(messages, pack, family) on data5/sft5_val.jsonl
rows (both device words, every pack in the file, chat rows without tools), with and without the generation prompt; tools
given as dicts, as names and as [1] (all 30 vs fmt4 pack True); and every system turn in spec/render_golden_v3.json `entries`.
The Lite template must contain `assistant\\n<think>\\n\\n</think>` (what LocalAgent.family() looks for).

    python spec/make_template_v3.py            # write both templates, then check them
    python spec/make_template_v3.py --check    # check the existing files only (run on the GPU box as RENDER_OK)
"""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen4"))
import fmt4 as F          # noqa: E402

OUT = {"4b": os.path.join(HERE, "chat_template_4b_v3.jinja"), "lite": os.path.join(HERE, "chat_template_lite_v3.jinja")}
SRC = {"4b": os.path.join(LAB, "export", "chat_template_4b.jinja"), "lite": os.path.join(LAB, "export", "chat_template_lite.jinja")}
OLD_HEAD = "{%- endif %}# Tools"          # the old literal block starts right after the system-content endif
OLD_TAIL = "</tool_call>{{- '<|im_end|>\\n' }}"


def jstr(s):
    """A Jinja single-quoted string literal for s (Jinja unescapes like Python's unicode-escape)."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def tool_block_jinja():
    lines = ",\n        ".join(f"({jstr(n)}, {jstr(F.TOOL_LINE[n])})" for n in F.TOOL_NAMES)
    return (
        "{%- endif %}\n"
        "    {%- set ns = namespace(names=[], bio='Face ID') %}\n"
        "    {%- if tools is iterable and tools is not string and tools is not mapping %}\n"
        "        {%- for t in tools %}\n"
        "            {%- if t is mapping and t.function is defined and t.function is mapping and t.function.name is defined %}\n"
        "                {%- set ns.names = ns.names + [t.function.name] %}\n"
        "            {%- elif t is mapping and t.name is defined %}\n"
        "                {%- set ns.names = ns.names + [t.name] %}\n"
        "            {%- elif t is string %}\n"
        "                {%- set ns.names = ns.names + [t] %}\n"
        "            {%- endif %}\n"
        "        {%- endfor %}\n"
        "    {%- endif %}\n"
        "    {%- if messages[0].role == 'system' and 'Touch ID' in messages[0].content and 'Face ID' not in messages[0].content %}\n"
        "        {%- set ns.bio = 'Touch ID' %}\n"
        "    {%- endif %}\n"
        "    {{- " + jstr(F.HEADER) + " }}\n"
        "    {%- for tool in [\n        " + lines + "\n    ] %}\n"
        "        {%- if not ns.names or tool[0] in ns.names %}\n"
        "            {{- '\\n' + (tool[1] | replace('Face ID', ns.bio)) }}\n"
        "        {%- endif %}\n"
        "    {%- endfor %}\n"
        "    {{- " + jstr(F.FOOTER) + " }}{{- '<|im_end|>\\n' }}")


def build(family):
    old = open(SRC[family]).read()
    i = old.index(OLD_HEAD)
    j = old.index(OLD_TAIL, i) + len(OLD_TAIL)
    assert old.count(OLD_HEAD) == 1 and old[i:j].count("# Tools") == 1, "old tool block not found exactly once"
    new = old[:i] + tool_block_jinja() + old[j:]
    title = "Hermit 1.3 Lite chat template" if family == "lite" else "Hermit 1.3 chat template"
    new = new.replace("Hermit 1.2 chat template", title + " (tools v3: 30 tools, the app's pack in registry order, device word)")
    new = new.replace("When tools are passed, the fixed Hermit tool block (hermit_tools.json) is used.",
                      "When tools are passed, the v3 tool block (spec/hermit_tools_v3.json) holds the passed tools (all 30 when they carry no names).")
    return new


def to_tpl(msgs):
    o = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in m["tool_calls"]]
        o.append(m)
    return o


def check(family, text):
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(text)
    tools_by = {t["function"]["name"]: t for t in F.TOOLS}
    pack_key = "pack_lite" if family == "lite" else "pack"
    n = bad = 0
    seen = {}
    for l in open(os.path.join(LAB, "data5", "sft5_val.jsonl")):
        ex = json.loads(l)
        meta = ex["meta"]
        if family == "lite" and "lite" not in (meta.get("family_ok") or []):
            continue
        pack = F.norm_pack(meta.get(pack_key) or [])
        msgs = ex["messages"]
        k = (tuple(pack), F.detect_bio(msgs))
        seen[k] = seen.get(k, 0) + 1
        variants = [("dicts", [tools_by[x] for x in pack])] if pack else [("none", None)]
        if pack and seen[k] <= 2:
            variants.append(("names", list(pack)))
        for how, tools in variants:
            n += 1
            a = tpl.render(messages=to_tpl(msgs), tools=tools, add_generation_prompt=False)
            b = F.render_text(msgs, pack, family)
            if a != b:
                bad += 1
                if bad <= 3:
                    p = next((q for q in range(min(len(a), len(b))) if a[q] != b[q]), min(len(a), len(b)))
                    print(f"MISMATCH {family} {how} {meta.get('source')} at {p}: tpl {a[p - 40:p + 60]!r} vs fmt4 {b[p - 40:p + 60]!r}")
        cut = max((i for i, m in enumerate(msgs) if m["role"] in ("user", "tool")), default=None)
        if cut is not None:
            sub = msgs[:cut + 1]; n += 1
            bad += tpl.render(messages=to_tpl(sub), tools=[tools_by[x] for x in pack] if pack else None, add_generation_prompt=True) != \
                F.render_text(sub, pack, family, add_generation_prompt=True)
        if seen[k] <= 1 and pack:                     # all 30 when the tools carry no names
            n += 1
            bad += tpl.render(messages=to_tpl(msgs), tools=[1], add_generation_prompt=False) != F.render_text(msgs, True, family)
    # golden system turns (spec/render_golden_v3.json entries: family, bio, pack tools, text = the system turn)
    g = json.load(open(os.path.join(HERE, "render_golden_v3.json")))
    gn = gbad = 0
    for e in g["entries"]:
        if e.get("family") not in (family, None):
            continue
        sysmsg = {"role": "system", "content": F.system_agent(e["bio"])}
        txt = tpl.render(messages=[sysmsg], tools=[tools_by[x] for x in e["tools"]], add_generation_prompt=False)
        gn += 1
        gbad += txt != e["text"]
    assert "assistant\\n<think>\\n\\n</think>" not in text or family == "lite"
    if family == "lite":
        assert "assistant\\n<think>\\n\\n</think>" in text, "LocalAgent.family() marker missing"
    print(f"{family}: {n} renders vs fmt4.render_text, mismatches {bad}; {len(seen)} (pack, device word) combinations; "
          f"golden system turns {gn}, mismatches {gbad}")
    return bad == 0 and gbad == 0


def main():
    only_check = "--check" in sys.argv
    ok = True
    for fam in ("4b", "lite"):
        if not only_check:
            text = build(fam)
            open(OUT[fam], "w").write(text)
            print("wrote", OUT[fam], len(text), "chars")
        ok &= check(fam, open(OUT[fam]).read())
    print("templates OK" if ok else "TEMPLATE MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
