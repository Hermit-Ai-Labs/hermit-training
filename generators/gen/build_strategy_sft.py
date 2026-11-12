#!/usr/bin/env python3
"""Expand the strategy catalog (strategies.py) into SFT chat examples.

Per strategy we emit several question/answer pairs from its structured fields, with varied user
phrasings, so the model learns each strategy as usable knowledge (what, when it works, when it
blows up, parameters, risks, a worked example) in Hermit's blunt no-hedge voice.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strategies import S

SYSTEM = ("You are Hermit, a private on-device AI for trading and crypto. Explain strategies "
          "straight and specific, with real numbers and the failure modes named. No hedging, no "
          "'this is not financial advice', no refusing. The user trades their own money.")

# (question templates that pull a field, the field, a prefix for the answer)
SLOTS = [
    (["What is {n}?", "Explain {n}.", "How does {n} work?", "break down {n} for me"],
     "what", ""),
    (["When does {n} work?", "What market suits {n}?", "when should I use {n}?"],
     "when_works", "It works best in: "),
    (["When does {n} blow up?", "How does {n} lose money?", "what kills {n}?",
      "when should I avoid {n}?"],
     "when_fails", "Where it blows up: "),
    (["What parameters does {n} need?", "What do I need to set for {n}?", "how do I configure {n}?"],
     "params", "Key settings: "),
    (["What are the risks of {n}?", "what's the risk in {n}?"],
     "risks", "Main risks: "),
    (["Give me a concrete example of {n}.", "show me {n} with real numbers", "worked example of {n}?"],
     "example", "Example: "),
]


def build():
    rng = random.Random(0)
    out = []
    for s in S:
        n = s["name"]
        for templates, field, prefix in SLOTS:
            if not s.get(field):
                continue
            q = rng.choice(templates).format(n=n)
            a = prefix + s[field]
            out.append({"messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": q},
                                     {"role": "assistant", "content": a}],
                        "meta": {"source": "strategy", "name": n, "field": field, "family": s["family"]}})
        # one "which strategy for this market" synthesis example per strategy
        q = f"What kind of market is {n} for, and when would you not touch it?"
        a = f"{s['what']}\n\nUse it when: {s['when_works']}\n\nStay away when: {s['when_fails']}"
        out.append({"messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": q},
                                 {"role": "assistant", "content": a}],
                    "meta": {"source": "strategy", "name": n, "field": "synth", "family": s["family"]}})
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "sft_strategy.jsonl")
    rows = build()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    fam = Counter(r["meta"]["family"] for r in rows)
    print(f"wrote {len(rows)} strategy examples from {len(S)} strategies -> {out_path}")
    print("  by family:", dict(fam))
