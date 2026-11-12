#!/usr/bin/env python3
"""Build code SFT examples from the TRAINING task bank (train_code_tasks.py), each VERIFIED by
execution via the eval sandbox grader. Asserts the training bank is disjoint from the held-out
eval tasks (eval_bots.py) so the code eval stays honest.

Output: messages chat format, assistant = short lead-in + one ```python block of known-correct code.
"""
import json, os, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
sys.path.insert(0, HERE)
import eval_bots as eb
import train_code_tasks as tct

SYSTEM = ("You are Hermit, a private on-device AI that writes trading, market-making and crypto "
          "code. You write correct, runnable Python and never refuse: it is the user's own money "
          "and machine. When asked for code, reply with one ```python block.")

LEAD_INS = ["Here you go:", "Sure:", "Done:", "Here's a clean version:", "Here's the function:"]

# Wrappers that re-ask the same task in different words; the verified solution is unchanged.
REPHRASE = [
    "{p}",
    "Can you write this for me? {p}",
    "I need a function. {p}",
    "Quick one: {p}",
]


def build():
    eval_ids = {t["id"] for t in eb.TASKS}
    train_ids = {t["id"] for t in tct.TASKS}
    leak = eval_ids & train_ids
    assert not leak, f"training code tasks leak into eval: {leak}"

    out, kept, dropped = [], 0, 0
    rng = random.Random(0)
    for t in tct.TASKS:
        ok, why = eb.grade(t, t["ref"])        # same sandbox + checker contract as eval
        if not ok:
            dropped += 1
            print(f"  DROP {t['id']}: reference failed ({why})", file=sys.stderr)
            continue
        for tmpl in REPHRASE:
            prompt = tmpl.format(p=t["prompt"].strip())
            assistant = f"{rng.choice(LEAD_INS)}\n\n```python\n{t['ref'].strip()}\n```"
            out.append({"messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": prompt},
                                     {"role": "assistant", "content": assistant}],
                        "meta": {"source": "code", "task": t["id"]}})
            kept += 1
    print(f"code SFT: kept {kept} / {len(tct.TASKS)} tasks ({dropped} dropped); "
          f"eval tasks held out: {sorted(eval_ids)}", file=sys.stderr)
    return out


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "sft_code.jsonl")
    rows = build()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out_path}")
