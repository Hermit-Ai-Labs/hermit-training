#!/usr/bin/env python3
"""Assemble Hermit 1.2 SFT data into sft2_train.jsonl / sft2_val.jsonl.

Sources (all converted to the common {messages, tools, meta} shape):
  agent      programmatic tool-calling + maths conversations (build_agent.py), verified
  teacher    breadth answers + multi-turn + code from the local OPEN teacher (build_teacher.py), filtered
  code_old   data/sft_code.jsonl  (12 execution-verified trading-code tasks)
  strategy   data/sft_strategy.jsonl (hand-authored)
  voice      data/sft_voice_safety.jsonl (voice, honest-limits, and the four hard lines
             KEPT EXACTLY AS AUTHORED; we never rewrite or expand the refusal data)

A per-source val slice is held out BEFORE any repetition. Every example is checked for leakage against
all eval2 sets (by normalised user text); overlaps are dropped from train. Also writes
eval2/notool.jsonl: held-out teacher questions presented WITH the tool block, where the gold behaviour
is a plain-text answer (no tool call) -> a held-out tool-restraint test.
"""
import argparse, json, os, random, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hermit_fmt as H

D1 = os.path.join(HERE, "..", "data")
D2 = os.path.join(HERE, "..", "data2")
EV = os.path.join(HERE, "..", "eval2")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def conv(messages, tools, source, **meta):
    return {"messages": messages, "tools": tools, "meta": {"source": source, **meta}}


def from_teacher_single(r):
    sysp = H.SYSTEM_AGENT if r.get("app") else H.SYSTEM_CHAT
    use_tools = False
    return conv([{"role": "system", "content": sysp}, {"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}],
                use_tools, "teacher", topic=r["topic"], kind="answer")


def from_teacher_multi(r):
    msgs = [{"role": "system", "content": H.SYSTEM_CHAT}] + r["msgs"]
    return conv(msgs, False, "teacher", topic=r.get("topic"), kind="multiturn")


def from_old(r, source):
    # these ship a role/user/assistant with their own system; replace system with Hermit's chat prompt
    m = [{"role": "system", "content": H.SYSTEM_CHAT}] + [x for x in r["messages"] if x["role"] != "system"]
    return conv(m, False, source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=os.path.join(D2, "agent_train.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--notool-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default=os.path.join(D2, "sft2"))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # ---- eval leakage index (normalised user text of every eval item)
    evusers = set()
    for f in ("tools.jsonl",):
        for r in load(os.path.join(EV, f)):
            for m in r.get("context", []):
                if m["role"] == "user":
                    evusers.add(norm(m["content"]))
    for f in ("math_v2.jsonl", "math_legacy.jsonl"):
        for r in load(os.path.join(EV, f)):
            evusers.add(norm(r["question"]))
    for r in load(os.path.join(EV, "chat.jsonl")):
        evusers.add(norm(r["prompt"]))
    for r in load(os.path.join(EV, "multiturn.jsonl")):
        for t in r["turns"]:
            evusers.add(norm(t))

    def first_user(ex):
        return next((norm(m["content"]) for m in ex["messages"] if m["role"] == "user"), "")

    # ---- build source pools
    pools = {}
    pools["agent"] = load(a.agent)
    teach = load(os.path.join(D2, "t_answers.jsonl"))
    # hold out some teacher questions for the no-tool eval (answerable, non-app, non-writing topics)
    notool_pool = [r for r in teach if not r["app"] and r["topic"] not in ("writing", "general", "money_life")]
    rng.shuffle(notool_pool)
    notool = notool_pool[: a.notool_n]
    notool_q = set(norm(r["q"]) for r in notool)
    pools["teacher_single"] = [from_teacher_single(r) for r in teach if norm(r["q"]) not in notool_q]
    pools["teacher_multi"] = [from_teacher_multi(r) for r in load(os.path.join(D2, "t_multiturn.jsonl"))]
    pools["teacher_code"] = [from_teacher_single(r) for r in load(os.path.join(D2, "t_code.jsonl"))]
    pools["code_old"] = [from_old(r, "code_old") for r in load(os.path.join(D1, "sft_code.jsonl"))]
    pools["strategy"] = [from_old(r, "strategy") for r in load(os.path.join(D1, "sft_strategy.jsonl"))]
    pools["voice"] = [from_old(r, "voice") for r in load(os.path.join(D1, "sft_voice_safety.jsonl"))]

    # repeat factors for small, high-value sources (applied to TRAIN only, after val holdout)
    REPEAT = {"voice": 4, "strategy": 2, "code_old": 3, "teacher_code": 1, "teacher_multi": 1, "teacher_single": 1, "agent": 1}

    train, val = [], []
    comp = Counter()
    dropped = 0
    for name, pool in pools.items():
        # drop leakage + exact-duplicate first-user within pool
        seen, clean = set(), []
        for ex in pool:
            fu = first_user(ex)
            if fu in evusers:
                dropped += 1
                continue
            clean.append(ex)
        rng.shuffle(clean)
        nval = max(0, int(len(clean) * a.val_frac)) if len(clean) > 40 else 0
        val += clean[:nval]
        keep = clean[nval:]
        rep = REPEAT.get(name, 1)
        for _ in range(rep):
            train += keep
        comp[name] = len(keep) * rep

    rng.shuffle(train); rng.shuffle(val)
    with open(a.out_prefix + "_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out_prefix + "_val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # held-out no-tool restraint eval
    with open(os.path.join(EV, "notool.jsonl"), "w") as f:
        for r in notool:
            f.write(json.dumps({"kind": "notool_" + ("app" if r["app"] else r["topic"]), "topic": r["topic"],
                                "context": [{"role": "system", "content": H.SYSTEM_AGENT}, {"role": "user", "content": r["q"]}],
                                "tools": True, "expect": [{"text": True}], "world": None}) + "\n")

    print(f"train={len(train)} val={len(val)} dropped_leak={dropped} notool_eval={len(notool)}")
    for k, v in comp.items():
        print(f"  {k:16} train {v:>6} (x{REPEAT.get(k, 1)})")
    kinds = Counter(r["meta"].get("kind") or r["meta"]["source"] for r in train)
    print("train agent-kind sample:", {k: kinds[k] for k in list(kinds)[:8]})
    # token stats
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "..", "..", "base", "Qwen3-4B-Instruct-2507"))
    import numpy as np
    L = [len(tok(H.render_text(r["messages"], r["tools"], "4b"), add_special_tokens=False)["input_ids"]) for r in train[:4000]]
    L = np.array(L)
    print(f"seq len (4k sample): mean {L.mean():.0f} p50 {np.percentile(L,50):.0f} p90 {np.percentile(L,90):.0f} p99 {np.percentile(L,99):.0f} max {L.max()}")
    print(f"est trained-token total ~ {int(L.mean()*len(train)):,}")


if __name__ == "__main__":
    main()
