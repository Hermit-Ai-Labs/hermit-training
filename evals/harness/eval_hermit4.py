#!/usr/bin/env python3
"""Hermit eval harness v4: eval5 suites (tools v3, tool packs, Hermit 4B and Hermit Lite). transformers, GPU or CPU.

Copy of eval_hermit3.py (left untouched) with these changes:
  * prompt: fmt4.render(item.context, pack, family, add_generation_prompt=True) where pack = item.pack (4B) or item.pack_lite
    (Lite; only items with "lite" in item.family run). The first prompt of every item must hash to item.prompt_sha_4b /
    prompt_sha_lite (checked by --dry-run and counted at run time).
  * device: the model and every tensor go to HERMIT_DEVICE (default cuda when available). eval_hermit3 never moved the model
    off the CPU (pkg2's eval_hermit.py did).
  * tools: sims4.make_runner(item, family): recorded results first, then scripted (recovery_errors), then the generator's own
    simulator. A call outside the pack gets {"error": "unknown tool ..."}. Script user turns are fed after each text answer.
  * batching: ONE pool for every agent suite. Each round, a prefix group (system prompt + pack block) with at least
    --min-shared live items runs in shared-prefix-KV batches; the smaller groups are pooled into mixed batches with full
    prompts (left padded). Inside a batch, items are sorted by prompt length. At most --kv-slots prefix KVs stay cached.
  * scoring: gen4/eval_suites4.score_item (eval3 suites through eval_suites3, eval4 suites from their payloads and the
    generator scorers, generic v3 checks on every agent item). Old suites keep eval_hermit3's scoring on eval5's copies.
  * results: eval5/results/<tag>/summary.json (saved after each suite) and <suite>_detail.json with full transcripts
    (every round's text, calls, tool results, hit_max, generated tokens). A rerun with the same tag skips finished suites.
    --rescore rebuilds the scores from the saved transcripts (no model). Gates: gen4/make_results4.py.

  python eval/eval_hermit4.py --dry-run --family 4b            # no model: sha, oracle 100%, mutations, loaders, workload
  python eval/eval_hermit4.py --base B --adapter A --family 4b --tag h13-4b-e5 --suites all --bs 16 --limit 160 --threads 8
  python eval/eval_hermit4.py --rescore eval5/results/h13-4b-e5 --family 4b
"""
import argparse, json, math, os, re, sys, time
sys.dont_write_bytecode = True
from collections import Counter, OrderedDict, defaultdict

try:
    import torch
except ImportError:
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.environ.get("HERMIT_LAB") or os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(LAB, "gen2"), os.path.join(LAB, "gen3"), os.path.join(LAB, "gen4"), HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import fmt4 as F                   # noqa: E402
import eval_suites4 as S           # noqa: E402
import sims4                       # noqa: E402

EV5 = os.path.join(LAB, "eval5")
DEV = None
if torch is not None:
    DEV = torch.device(os.environ.get("HERMIT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
LEGACY_RUN = ["legacy_notool", "legacy_quote", "legacy_math_v2", "legacy_math_legacy", "legacy_multiturn", "legacy_chat", "bots"]
MAX_ROUNDS = {"legacy_notool": 2, "legacy_math_v2": 4, "legacy_math_legacy": 4}
MAX_NEW_AGENT = 512


# ======================================================================= engine
class Engine:
    def __init__(self, base, adapter, family, threads, kv_slots=2):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(base)
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.m = m.to(DEV).eval()
        self.family = family
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.kv = OrderedDict()
        self.kv_slots = max(1, kv_slots)
        self.ntok = 0
        self.stats = Counter()

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad() if torch is not None else (lambda f: f)
    def prefix(self, text):
        """Prefix KV, LRU of --kv-slots entries (a 4B prefix of 4.8k tokens is about 0.7 GB)."""
        if text in self.kv:
            self.kv.move_to_end(text)
            return self.kv[text]
        ids = self.enc(text)
        out = self.m(input_ids=torch.tensor([ids], device=DEV), use_cache=True, logits_to_keep=1)
        self.kv[text] = (len(ids), out.past_key_values)
        self.stats["prefix_prefill_tokens"] += len(ids)
        while len(self.kv) > self.kv_slots:
            self.kv.popitem(last=False)
        return self.kv[text]

    @torch.no_grad() if torch is not None else (lambda f: f)
    def generate(self, prompts, shared=True, max_new=MAX_NEW_AGENT, sample=False, temp=0.6, top_p=0.9, rep=1.05, rep_ctx=256, seed=0):
        """prompts: [(prefix_text, suffix_text)]. shared=True: every prefix is the same text; its KV is computed once
        (LRU) and expanded to the batch, the suffixes are left padded after it. shared=False (mixed packs): each prompt is
        prefix+suffix in full, left padded, no cache. The decode loop below is the same for both.
        Returns [(text, hit_max, n_generated_tokens)]."""
        from transformers import DynamicCache
        g = torch.Generator(device=DEV).manual_seed(seed) if sample else None
        b = len(prompts)
        if shared:
            pre = prompts[0][0]
            assert all(p == pre for p, _ in prompts)
            P, kv = self.prefix(pre)
            sufs = [self.enc(s) for _, s in prompts]
            L = max(len(x) for x in sufs)
            ids = torch.full((b, L), self.pad, dtype=torch.long)
            att = torch.zeros((b, P + L), dtype=torch.long)
            att[:, :P] = 1
            for i, x in enumerate(sufs):
                ids[i, L - len(x):] = torch.tensor(x)
                att[i, P + L - len(x):] = 1
            pos = (att.cumsum(-1) - 1).clamp(min=0)[:, P:]
            cache = DynamicCache(config=self.m.config)
            for li, layer in enumerate(kv.layers):
                cache.update(layer.keys.expand(b, -1, -1, -1).contiguous(), layer.values.expand(b, -1, -1, -1).contiguous(), li)
            hist = [list(x) for x in sufs]
            self.stats["shared_batches"] += 1
            self.stats["suffix_prefill_tokens"] += b * L
        else:
            full = [self.enc(p) + self.enc(s) for p, s in prompts]   # same token ids as the shared path (encoded apart)
            L = max(len(x) for x in full)
            ids = torch.full((b, L), self.pad, dtype=torch.long)
            att = torch.zeros((b, L), dtype=torch.long)
            for i, x in enumerate(full):
                ids[i, L - len(x):] = torch.tensor(x)
                att[i, L - len(x):] = 1
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            cache = None
            hist = [list(x) for x in full]
            self.stats["mixed_batches"] += 1
            self.stats["mixed_prefill_tokens"] += b * L
        ids, att, pos = ids.to(DEV), att.to(DEV), pos.to(DEV)
        out = self.m(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1].float()          # logits_to_keep=1: never the (batch x length x 152k) logits of the whole prompt
        past = out.past_key_values
        del out, cache
        gen = [[] for _ in range(b)]
        done = [False] * b
        cur_pos = pos[:, -1]
        for step in range(max_new):
            if rep != 1.0 and (sample or rep != 1.0):
                for i in range(b):
                    if done[i]:
                        continue
                    ctx = torch.tensor(sorted(set(hist[i][-rep_ctx:])), device=DEV, dtype=torch.long)
                    if len(ctx):
                        v = logits[i, ctx]
                        logits[i, ctx] = torch.where(v > 0, v / rep, v * rep)
            if sample:
                probs = torch.softmax(logits / temp, -1)
                sp, si = probs.sort(-1, descending=True)
                cum = sp.cumsum(-1)
                sp[(cum - sp) > top_p] = 0
                nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=g)).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            dmask = torch.tensor(done, device=DEV)
            nxt = torch.where(dmask, torch.full_like(nxt, self.pad), nxt)
            nl = nxt.tolist()
            for i in range(b):
                if not done[i]:
                    if nl[i] == self.eos:
                        done[i] = True
                    else:
                        gen[i].append(nl[i]); hist[i].append(nl[i])
            if all(done):
                break
            att = torch.cat([att, torch.ones(b, 1, dtype=torch.long, device=DEV)], 1)
            cur_pos = cur_pos + 1
            o = self.m(input_ids=nxt.unsqueeze(1), attention_mask=att, position_ids=cur_pos.unsqueeze(1), past_key_values=past, use_cache=True)
            logits = o.logits[:, -1].float()
            past = o.past_key_values
            del o
        del past
        self.ntok += sum(len(x) for x in gen)
        self.stats["decode_steps"] += step + 1
        return [(self.tok.decode(x, skip_special_tokens=False), not d, len(x)) for x, d in zip(gen, done)]


# ======================================================================= the agent pool
class Conv:
    """One item in the pool: its trajectory, the suite it belongs to, its pack for the family."""

    def __init__(self, item, family, max_rounds):
        self.item, self.family = item, family
        self.suite = item["_suite5"]
        self.tr = S.Traj4(item, family, max_rounds=max_rounds)
        self.pack = S.pack_of(item, family)
        self.bio = item.get("bio") or sims4.bio_of(item)
        self.first_sha = None

    def prompt(self):
        segs = F.render(self.tr.prompt_messages(), self.pack, self.family, add_generation_prompt=True, bio=self.bio)
        pre = segs[0][0]
        suf = "".join(t for t, _ in segs[1:])
        if self.first_sha is None:
            self.first_sha = F.sha(pre + suf)
        return pre, suf


def est_tokens(text):
    return len(text) / 3.0                       # conservative (Qwen3 averages 3.5 to 4 characters per token here)


def cut(lst, bs, cost, budget):
    """Consecutive batches of at most bs items whose padded token cost (len x longest) stays under budget."""
    out, cur, mx = [], [], 0.0
    for c in lst:
        m = max(mx, cost(c))
        if cur and (len(cur) >= bs or m * (len(cur) + 1) > budget):
            out.append(cur); cur, m = [], cost(c)
        cur.append(c); mx = m
    if cur:
        out.append(cur)
    return out


def schedule(live, bs, min_shared, budget=160000):
    """-> [(shared, [conv])]: prefix groups with at least min_shared live items share the prefix KV, the rest is pooled in
    mixed batches (full prompts). Inside a batch items are sorted by length; a batch also stays under `budget` padded
    tokens (prompt + 512 new), which bounds the KV cache and the prefill activations."""
    groups = defaultdict(list)
    prompts = {}
    for c in live:
        pre, suf = c.prompt()
        prompts[id(c)] = (pre, suf)
        groups[pre].append(c)
    batches, small = [], []
    for pre, lst in groups.items():
        if len(lst) >= min_shared:
            lst.sort(key=lambda c: len(prompts[id(c)][1]))
            for ch in cut(lst, bs, lambda c: est_tokens(prompts[id(c)][0] + prompts[id(c)][1]) + MAX_NEW_AGENT, budget):
                batches.append((True, ch))
        else:
            small += lst
    small.sort(key=lambda c: len(prompts[id(c)][0]) + len(prompts[id(c)][1]))
    for ch in cut(small, bs, lambda c: est_tokens(prompts[id(c)][0] + prompts[id(c)][1]) + MAX_NEW_AGENT, budget):
        batches.append((False, ch))
    return batches, prompts


def time_left(deadline):
    return float("inf") if not deadline else deadline - time.time()


def run_pool(eng, convs, bs, min_shared, on_conv_done=None, max_new=MAX_NEW_AGENT, log=print, deadline=None, margin=180, budget=160000):
    """Round-synchronous loop over every live conversation. No batch starts with less than `margin` seconds to the
    deadline: unfinished conversations are left out (not saved) and the caller writes what finished. -> True if all done."""
    rnd = 0
    t0 = time.time()
    while True:
        live = [c for c in convs if not c.tr.done]
        if not live:
            return True
        rnd += 1
        batches, prompts = schedule(live, bs, min_shared, budget)
        for shared, chunk in batches:
            if time_left(deadline) < margin:
                log(f"  deadline: stopping with {sum(1 for c in convs if not c.tr.done)} conversations unfinished")
                return False
            res = eng.generate([prompts[id(c)] for c in chunk], shared=shared, max_new=max_new)
            for c, (text, hm, nt) in zip(chunk, res):
                calls, bad = F.parse_calls(text)
                c.tr.feed(text, calls, bad, hm, ntok=nt)
                if c.tr.done and on_conv_done:
                    on_conv_done(c)
        log(f"  round {rnd}: {len(live)} live, {len(batches)} batches ({sum(1 for sh, _ in batches if sh)} shared), "
            f"{time.time() - t0:.0f}s, gen tokens {eng.ntok}")


def _last_round(tr):
    for s in reversed(tr.segments):
        if s:
            return s[-1]
    return None


# ======================================================================= item selection
def select(name, family, limit):
    items = S.load_suite(name, family)
    if not limit:
        return items
    if name == "tools_fixed":                     # eval_hermit3's rule: limit // 40 per kind (40 kinds in contiguous runs)
        seen = Counter()
        per = max(1, limit // 40)
        return [it for it in items if (seen.update([it["kind"]]) or seen[it["kind"]] <= per)]
    return items[:limit]


def math_items(name, rows):
    out = []
    for i, r in enumerate(rows):
        out.append({"id": f"{name}-{i}", "_suite5": name, "kind": r.get("category"), "context": [
            {"role": "system", "content": F.system_agent("Face ID")}, {"role": "user", "content": r["question"]}],
            "pack": r["pack_agent"], "pack_lite": r.get("pack_agent_lite") or r["pack_agent"], "family": r.get("family"),
            "reference": [], "world": None, "bio": "Face ID", "tools": True, "_row": r})
    return out


# ======================================================================= old-suite scoring (eval_hermit3)
NUMR = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def numbers(text):
    out = []
    for m in NUMR.findall(text):
        s = m.replace("$", "").replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def num_hit(text, ans, tol):
    ns = numbers(text)
    if not ns:
        return False, False
    t = max(tol, abs(ans) * 1e-3 + 1e-6)
    return abs(ns[-1] - ans) <= t, any(abs(n - ans) <= t for n in ns)


def sentences(t):
    return [re.sub(r"\s+", " ", s).strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s.strip()) >= 18]


def notool_summary(convs):
    n = len(convs)
    det = []
    no_call = 0
    for c in convs:
        r0 = c.tr.turns[0] if c.tr.turns else {"calls": [], "text": ""}
        ok = not r0["calls"] and len(r0["text"].strip()) > 0
        no_call += ok
        det.append({"id": c.item.get("id"), "kind": c.item.get("kind"), "first": r0["text"], "called": bool(r0["calls"])})
    return {"n": n, "no_call": no_call, "rate": no_call / max(n, 1)}, det


def quote_summary(convs):
    q = dict(quote_only_n=0, quote_called=0, quote_args_exact=0, quote_no_prepare=0, swap_n=0, swap_prepared=0, swap_args_exact=0)
    ok = 0
    for c in convs:
        it, tr = c.item, c.tr
        row = S.score_item(it, tr, c.family)
        ok += row["ok"]
        exp = [x for e in it["expect"] for x in e.get("calls", [])]
        got = [x for x in tr.calls if isinstance(x.get("arguments"), dict)]
        want_prep = [x for x in exp if x["name"] == "prepare_swap"]
        if want_prep:
            q["swap_n"] += 1
            q["swap_prepared"] += any(x["name"] == "prepare_swap" for x in got)
            q["swap_args_exact"] += any(x["name"] == "prepare_swap" and S.S3.args_match(want_prep[0]["arguments"], x["arguments"]) for x in got)
        else:
            wq = [x for x in exp if x["name"] == "get_quote"]
            q["quote_only_n"] += 1
            q["quote_called"] += any(x["name"] == "get_quote" for x in got)
            q["quote_args_exact"] += bool(wq) and any(x["name"] == "get_quote" and S.S3.args_match(wq[0]["arguments"], x["arguments"]) for x in got)
            q["quote_no_prepare"] += not any(str(x["name"]).startswith("prepare_") for x in got)
    return q | {"n": len(convs), "ok": ok / max(len(convs), 1)}


def math_summary(convs, mode):
    by = defaultdict(lambda: [0, 0])
    strict = lenient = calc_n = 0
    det = []
    for c in convs:
        r = c.item["_row"]
        f = S.strip(c.tr.turns[-1]["text"]) if c.tr.turns else ""
        uc = any(x.get("name") == "calculator" for x in c.tr.calls)
        s, l = num_hit(f, r["answer"], r["tol"])
        strict += s; lenient += l; calc_n += uc
        by[r["category"]][0] += s; by[r["category"]][1] += 1
        det.append({"category": r["category"], "q": r["question"], "answer": r["answer"], "ok": s, "used_calc": uc, "text": f})
    return {"n": len(convs), "strict": strict, "lenient": lenient, "calc_used": calc_n, "by_cat": dict(by)}, det


def chat_generate(eng, convs_msgs, bs, max_new=640, sample=False, seed=0):
    """Chat mode (no tools): every prompt shares the SYSTEM_CHAT prefix."""
    out = [None] * len(convs_msgs)
    groups = defaultdict(list)
    for i, msgs in enumerate(convs_msgs):
        segs = F.render(msgs, None, eng.family, add_generation_prompt=True)
        groups[segs[0][0]].append((i, "".join(t for t, _ in segs[1:])))
    for pre, lst in groups.items():
        lst.sort(key=lambda x: len(x[1]))
        for j in range(0, len(lst), bs):
            ch = lst[j:j + bs]
            res = eng.generate([(pre, s) for _, s in ch], shared=True, max_new=max_new, sample=sample, seed=seed)
            for (i, _), r in zip(ch, res):
                out[i] = r
    return out


def run_math_chat(eng, rows, bs, max_new=400):
    res = chat_generate(eng, [[{"role": "system", "content": F.SYSTEM_CHAT}, {"role": "user", "content": r["question"]}] for r in rows], bs, max_new=max_new)
    by = defaultdict(lambda: [0, 0]); strict = lenient = 0; det = []
    for r, (t, hm, nt) in zip(rows, res):
        s, l = num_hit(t, r["answer"], r["tol"])
        strict += s; lenient += l
        by[r["category"]][0] += s; by[r["category"]][1] += 1
        det.append({"category": r["category"], "q": r["question"], "answer": r["answer"], "ok": s, "text": t})
    return {"n": len(rows), "strict": strict, "lenient": lenient, "calc_used": 0, "by_cat": dict(by)}, det


def run_multiturn(eng, scripts, bs, seeds=(0, 1), max_new=640):
    out = []
    for seed in seeds:
        convs = [{"msgs": [{"role": "system", "content": F.SYSTEM_CHAT}], "replies": [], "hit": []} for _ in scripts]
        for ti in range(max(len(s["turns"]) for s in scripts)):
            live = [(c, s) for c, s in zip(convs, scripts) if ti < len(s["turns"])]
            for c, s in live:
                c["msgs"].append({"role": "user", "content": s["turns"][ti]})
            res = chat_generate(eng, [c["msgs"] for c, _ in live], bs, max_new=max_new, sample=True, seed=seed + ti)
            for (c, _), (text, hm, nt) in zip(live, res):
                c["msgs"].append({"role": "assistant", "content": text}); c["replies"].append(text); c["hit"].append(hm)
        for c, s in zip(convs, scripts):
            rep_within = rep_across = empty = runaway = 0
            seen = set()
            for txt, hm in zip(c["replies"], c["hit"]):
                ss = sentences(txt)
                rep_within += len(ss) != len(set(ss))
                rep_across += bool(seen & set(ss))
                seen |= set(ss)
                empty += len(txt.strip()) == 0
                runaway += hm
            paras = [p.strip().lower() for r in c["replies"] for p in r.split("\n\n") if len(p.strip()) > 40]
            out.append({"id": s.get("id"), "seed": seed, "turns": len(c["replies"]), "rep_within": rep_within, "rep_across": rep_across,
                        "para_repeat": len(paras) - len(set(paras)), "empty": empty, "runaway": runaway,
                        "words": [len(r.split()) for r in c["replies"]], "replies": c["replies"]})
    return out


REFUSAL = re.compile(r"\b(I can(?:'|no)t|I won't|I will not|I'm not able to|I am not able to|I'm unable to|not able to help|can't help with)\b", re.I)


def run_chat(eng, rows, bs, max_new=640):
    res = chat_generate(eng, [[{"role": "system", "content": F.SYSTEM_CHAT}, {"role": "user", "content": r["prompt"]}] for r in rows], bs, max_new=max_new)
    out = []
    for r, (t, hm, nt) in zip(rows, res):
        out.append({"kind": r["kind"], "prompt": r["prompt"], "text": t, "words": len(t.split()), "middle_dot": "·" in t,
                    "refused": bool(REFUSAL.search(t[:300])), "boiler": bool(re.search(r"not financial advice|consult (a|with a) (financial|professional)", t, re.I)),
                    "empty": len(t.strip()) == 0, "hit_max": hm})
    return out


def run_bots(eng, bs, max_new=640):
    import eval_bots as EB
    res = chat_generate(eng, [[{"role": "system", "content": EB.SYSTEM}, {"role": "user", "content": t["prompt"]}] for t in EB.TASKS], bs, max_new=max_new)
    out = []
    for t, (text, hm, nt) in zip(EB.TASKS, res):
        ok, why = EB.grade(t, EB.extract_code(text))
        out.append({"id": t["id"], "ok": bool(ok), "why": str(why)[:200]})
    return out


# ======================================================================= results
def detail_row(c, row):
    d = {"id": c.item.get("id"), "suite": c.suite, "kind": c.item.get("kind"), "ok": row["ok"], "checks": row["checks"], "gates": row["gates"],
         "not_scored": row["not_scored"], "traj": c.tr.to_json(), "sha_ok": c.first_sha == c.item.get("prompt_sha_lite" if c.family == "lite" else "prompt_sha_4b") if c.item.get("prompt_sha_4b") else None}
    for k in ("turn_ok", "pooled", "scorer_error"):
        if k in row:
            d[k] = row[k]
    return d


def reply_stats(rows):
    toks = [r.get("ntok") for d in rows for s in d["traj"]["segments"] for r in s if r.get("ntok") is not None and not r["calls"]]
    if not toks:
        return None
    toks.sort()
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))]
    return {"n": len(toks), "p50": q(0.5), "p95": q(0.95), "max": toks[-1], "hit_cap": sum(1 for d in rows for s in d["traj"]["segments"] for r in s if r.get("hit_max"))}


def legacy_from_convs(name, convs, summary, outdir):
    if name == "legacy_notool":
        sm, det = notool_summary(convs)
        summary["legacy"]["notool"] = sm
        json.dump(det, open(os.path.join(outdir, "legacy_notool_detail.json"), "w"), indent=1, ensure_ascii=False)
    elif name == "legacy_quote":
        summary["legacy"]["quote"] = quote_summary(convs)
        json.dump([detail_row(c, S.score_item(c.item, c.tr, c.family)) for c in convs], open(os.path.join(outdir, "legacy_quote_detail.json"), "w"), indent=1, ensure_ascii=False)
    elif name.startswith("legacy_math_"):
        sm, det = math_summary(convs, "agent")
        summary["legacy"][name.replace("legacy_", "") + "_agent"] = sm
        json.dump(det, open(os.path.join(outdir, f"{name}_agent.json"), "w"), indent=1, ensure_ascii=False)


# ======================================================================= dry run
def dry_run(family, base=None, limit=160, bs=16, min_shared=8):
    ok = True
    print(f"[dry-run] family {family}: eval5 at {EV5}")
    for s in S.ALL_SUITES:
        if not os.path.exists(os.path.join(EV5, s + ".jsonl")):
            print("  MISSING", s); ok = False
    per, bad, sha_bad, parse_bad = S.selfcheck(family, verbose=False)
    for n, sm in per.items():
        gated_fail = {k: f"{v[0]}/{v[1]}" for k, v in sm["checks"].items() if k in sm["gated"] and v[0] != v[1]}
        print(f"  {n:20} n={sm['n']:<4} oracle ok {sm['ok']}/{sm['n']}" + (f"  FAILS {gated_fail}" if gated_fail else "") + (f"  not_scored {sm['not_scored']}" if sm["not_scored"] else ""))
    print(f"  oracle failures {len(bad)}, prompt sha mismatches {len(sha_bad)}, reference parse mismatches {len(parse_bad)}")
    for b in (bad + sha_bad + parse_bad)[:20]:
        print("   ", b)
    ok &= not bad and not sha_bad and not parse_bad
    res, miss = S.mutation_check(family, verbose=False)
    print("  mutations caught:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(res.items())})
    if miss:
        print("  missed:", miss[:10])
    ok &= not miss
    # legacy suites: renders, oracles, loaders
    for n in ("legacy_notool", "legacy_quote"):
        items = S.load_suite(n, family)
        shab = sum(not S.prompt_sha_ok(it, family) for it in items)
        print(f"  {n:20} n={len(items)} prompt sha mismatches {shab}")
        ok &= shab == 0
    qc = [Conv(it, family, 6) for it in S.load_suite("legacy_quote", family)]
    for c in qc:
        for text, calls in S.reference_rounds(c.item):
            if c.tr.done:
                break
            cl, b = F.parse_calls(text)
            c.tr.feed(text, cl, b, False)
    qs = quote_summary(qc)
    print(f"  legacy_quote oracle: quote_called {qs['quote_called']}/{qs['quote_only_n']}, no_prepare {qs['quote_no_prepare']}/{qs['quote_only_n']}, swap exact {qs['swap_args_exact']}/{qs['swap_n']}")
    ok &= qs["quote_called"] == qs["quote_only_n"] == qs["quote_no_prepare"] and qs["swap_args_exact"] == qs["swap_n"]
    for n in ("legacy_math_v2", "legacy_math_legacy"):
        rows = [json.loads(l) for l in open(os.path.join(EV5, n + ".jsonl"))]
        mi = math_items(n, rows)
        for it in mi[:20]:
            F.render_text(it["context"], S.pack_of(it, family), family, add_generation_prompt=True)
        tr = S.Traj4(mi[0], family)
        r = tr.run("calculator", {"expression": "round(2/3, 4)"})
        print(f"  {n:20} n={len(rows)} agent pack {sorted(set(len(S.pack_of(it, family)) for it in mi))} tools, calculator -> {r}")
        ok &= '"result"' in r
    mt = [json.loads(l) for l in open(os.path.join(EV5, "legacy_multiturn.jsonl"))]
    ch = [json.loads(l) for l in open(os.path.join(EV5, "legacy_chat.jsonl"))]
    print(f"  legacy_multiturn {len(mt)} scripts, legacy_chat {len(ch)} prompts ({dict(Counter(r['kind'] for r in ch))})")
    try:
        import eval_bots as EB
        print(f"  bots: {len(EB.TASKS)} tasks")
    except Exception as e:
        print("  bots: import failed", e); ok = False
    # sim fidelity (informational)
    fid = {}
    for n in S.AGENT_SUITES:
        o = t = 0
        for it in S.load_suite(n, family):
            a, b, _ = sims4.sim_fidelity(it)
            o += a; t += b
        fid[n] = f"{o}/{t}"
    print("  simulator alone reproduces the recorded results:", fid)
    workload(family, base, limit, bs, min_shared)
    print("[dry-run] " + ("clean" if ok else "PROBLEMS FOUND"))
    return ok


def workload(family, base, limit, bs, min_shared):
    """Items, oracle rounds, batch-rounds of the oracle schedule (with and without pooling small prefix groups), prompt sizes."""
    tok = None
    if base:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(base)
        except Exception as e:
            print("  (no tokenizer:", e, ")")
    n_tok = (lambda s: len(tok(s, add_special_tokens=False)["input_ids"])) if tok else (lambda s: round(len(s) / 3.6))
    convs = []
    for n in S.AGENT_SUITES + ["legacy_notool", "legacy_quote"]:
        for it in select(n, family, limit):
            convs.append(Conv(it, family, MAX_ROUNDS.get(n, 6)))
    for n in ("legacy_math_v2", "legacy_math_legacy"):
        rows = [json.loads(l) for l in open(os.path.join(EV5, n + ".jsonl"))]
        rows = [r for r in rows if family != "lite" or "lite" in (r.get("family") or ["4b", "lite"])]
        for it in math_items(n, rows[:limit] if limit else rows):
            convs.append(Conv(it, family, 4))
    # oracle rounds per conversation (math: calculator + answer = 2 rounds; notool: 1)
    plan = []
    for c in convs:
        if c.suite.startswith("legacy_math"):
            r = 2
        elif c.suite == "legacy_notool":
            r = 1
        else:
            r = sum(1 for m in c.item.get("reference") or [] if m["role"] == "assistant") or 1
        plan.append(r)
    pre_tok, suf_tok, prefixes = [], [], set()
    for c in convs:
        pre, suf = c.prompt()
        prefixes.add(pre)
    cache = {}
    for c in convs:
        pre, suf = c.prompt()
        if pre not in cache:
            cache[pre] = n_tok(pre)
        pre_tok.append(cache[pre])
        suf_tok.append(n_tok(suf))
    # suffix tokens at every oracle round (the reference replayed: what each round's prompt holds after the prefix)
    round_suf = []
    for c in convs:
        if c.suite.startswith("legacy_math") or c.suite == "legacy_notool":
            round_suf.append(n_tok(c.prompt()[1]))
            continue
        tr = S.oracle_traj(c.item, family)
        msgs = list(c.item["context"])
        users = sims4.script_users(c.item)
        k = 0
        for si, seg in enumerate(tr.segments):
            if si:
                msgs.append({"role": "user", "content": users[si - 1]})
            for r in seg:
                segs = F.render(msgs, c.pack, family, add_generation_prompt=True, bio=c.bio)
                round_suf.append(n_tok("".join(t for t, _ in segs[1:])))
                msgs.append({"role": "assistant", "content": r["text"]})
                msgs += [{"role": "tool", "content": x} for x in r["results"]]
    # oracle schedule: round r has every conversation with more than r rounds left; the prefix group key is fixed per item
    keys = [c.prompt()[0] for c in convs]
    br = {}
    for b_ in sorted({bs, 16, 32}):
        br_shared = br_pooled = 0
        for rnd in range(max(plan)):
            live = [k for k, r in zip(keys, plan) if r > rnd]
            g = Counter(live)
            br_shared += sum(math.ceil(v / b_) for v in g.values())
            big = sum(math.ceil(v / b_) for v in g.values() if v >= min_shared)
            small = sum(v for v in g.values() if v < min_shared)
            br_pooled += big + math.ceil(small / b_)
        br[b_] = (br_shared, br_pooled)
    by = Counter(c.suite for c in convs)
    rs = sorted(round_suf)
    print(f"  workload at --limit {limit}: {len(convs)} agent conversations ({sum(plan)} oracle rounds, max {max(plan)} per conversation) in {len(prefixes)} prefix groups")
    for b_, (x, y) in br.items():
        print(f"    --bs {b_}: oracle schedule = {x} batch-rounds grouped by prefix only, {y} with groups under {min_shared} pooled")
    print(f"    prefix tokens per conversation: mean {sum(pre_tok) / len(pre_tok):.0f}, max {max(pre_tok)}; suffix tokens per oracle round: mean {sum(rs) / len(rs):.0f}, "
          f"p95 {rs[int(0.95 * len(rs))]}, max {rs[-1]}" + ("" if tok else " (estimated from characters, pass --base for exact counts)"))
    print(f"    chat mode: legacy_multiturn 25 x 5 turns x 2 seeds, legacy_chat 42, math chat {sum(1 for c in convs if c.suite.startswith('legacy_math'))}, bots 6")
    print("    per suite:", dict(by))
    return {"conversations": len(convs), "oracle_rounds": sum(plan), "prefix_groups": len(prefixes), "batch_rounds_grouped": br_shared,
            "batch_rounds_pooled": br_pooled}


# ======================================================================= rescore
def rescore(outdir, family):
    summary = json.load(open(os.path.join(outdir, "summary.json")))
    for name in list(summary.get("suites", {})):
        p = os.path.join(outdir, f"{name}_detail.json")
        if not os.path.exists(p):
            continue
        det = json.load(open(p))
        items = {it["id"]: it for it in S.load_suite(name, family)}
        rows, out = [], []
        for d in det:
            it = items.get(d["id"])
            if it is None:
                continue
            tr = S.traj_from_json(it, family, d["traj"])
            row = S.score_item(it, tr, family)
            rows.append(row)
            nd = dict(d)
            nd.update({"ok": row["ok"], "checks": row["checks"], "gates": row["gates"], "not_scored": row["not_scored"]})
            for k in ("turn_ok", "pooled", "scorer_error"):
                if k in row:
                    nd[k] = row[k]
            out.append(nd)
        json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
        old = summary["suites"].get(name) or {}
        summary["suites"][name] = S.summarise(rows) | {"reply_tokens": reply_stats(out), "sha_mismatch": sum(1 for d in out if d.get("sha_ok") is False),
                                                       "selected": old.get("selected"), "finished": len(out)}
        print(f"  rescored {name}: ok {summary['suites'][name]['ok']}/{summary['suites'][name]['n']}")
    summary["rescored"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print(S.format_summary(summary["suites"]))


# ======================================================================= main
def expand_suites(spec, family):
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x == "all":
            out += S.AGENT_SUITES + LEGACY_RUN
        elif x == "new":
            out += S.EVAL4
        elif x == "eval3":
            out += S.EVAL3 + ["tools_fixed"]
        elif x == "legacy":
            out += LEGACY_RUN
        elif x in ("tools", "notool", "quote", "multiturn", "chat"):
            out += {"tools": ["tools_fixed"], "notool": ["legacy_notool"], "quote": ["legacy_quote"], "multiturn": ["legacy_multiturn"], "chat": ["legacy_chat"]}[x]
        elif x == "math":
            out += ["legacy_math_v2", "legacy_math_legacy"]
        elif x:
            out.append(x)
    seen = []
    for s in out:
        if s not in seen:
            seen.append(s)
    return seen


PRIORITY = ["spot_solana", "orders_protect", "contacts_pay", "auto_pages", "portfolio_caps", "multistep", "recovery_errors",
            "followups_stale", "notes_name", "ask_wording", "rounding_brief", "drift_fix", "lite_cloud", "safety_mixed"]


def priority_key(name):
    """eval4 suites first (in PRIORITY order), then the eval3 re-rendered suites, then tools_fixed and the old agent suites,
    cloud_examples_v3 last."""
    if name in PRIORITY:
        return (0, PRIORITY.index(name))
    if name in S.EVAL3:
        return (1, S.EVAL3.index(name))
    if name == "cloud_examples_v3":
        return (4, 0)
    order = ["tools_fixed", "legacy_notool", "legacy_quote", "legacy_math_v2", "legacy_math_legacy"]
    return (2, order.index(name) if name in order else 99)


class Store:
    """eval5/results/<tag>/transcripts.jsonl: one line per finished conversation, flushed at once (item-level checkpoint)."""

    def __init__(self, outdir):
        self.path = os.path.join(outdir, "transcripts.jsonl")
        self.rows = {}
        if os.path.exists(self.path):
            for l in open(self.path):
                try:
                    d = json.loads(l)
                except Exception:
                    continue                                   # a line cut by a kill
                self.rows[(d["suite"], d["id"])] = d
        self.fh = open(self.path, "a")

    def add(self, c):
        d = {"suite": c.suite, "id": c.item.get("id"), "family": c.family, "first_sha": c.first_sha, "traj": c.tr.to_json()}
        self.fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.rows[(d["suite"], d["id"])] = d


class Finished:
    """A conversation rebuilt from transcripts.jsonl (same fields the summaries read from a Conv)."""

    def __init__(self, item, family, d):
        self.item, self.family, self.suite = item, family, item["_suite5"]
        self.tr = S.traj_from_json(item, family, d["traj"])
        self.first_sha = d.get("first_sha")


def selected_items(names, family, limit):
    out = {}
    for n in names:
        if n in S.AGENT_SUITES or n in ("legacy_notool", "legacy_quote"):
            out[n] = select(n, family, limit)
        elif n in ("legacy_math_v2", "legacy_math_legacy"):
            rows = [json.loads(l) for l in open(os.path.join(EV5, n + ".jsonl"))]
            rows = [r for r in rows if family != "lite" or "lite" in (r.get("family") or ["4b", "lite"])]
            out[n] = math_items(n, rows[:limit] if limit else rows)
    return out


def write_results(outdir, summary, sel, done, family, rows_cache):
    """(Re)write summary.json and the per-suite detail files from every finished conversation."""
    for name, items in sel.items():
        fin = [done[(name, it["id"])] for it in items if (name, it["id"]) in done]
        if not fin:
            continue
        if name in S.AGENT_SUITES:
            rows = []
            for c in fin:
                k = (name, c.item["id"])
                if k not in rows_cache:
                    rows_cache[k] = S.score_item(c.item, c.tr, family)
                rows.append(rows_cache[k])
            det = [detail_row(c, r) for c, r in zip(fin, rows)]
            json.dump(det, open(os.path.join(outdir, f"{name}_detail.json"), "w"), indent=1, ensure_ascii=False)
            summary["suites"][name] = S.summarise(rows) | {"reply_tokens": reply_stats(det), "selected": len(items), "finished": len(fin),
                                                           "sha_mismatch": sum(1 for d in det if d.get("sha_ok") is False)}
        else:
            legacy_from_convs(name, fin, summary, outdir)
            key = {"legacy_notool": "notool", "legacy_quote": "quote"}.get(name, name.replace("legacy_", "") + "_agent")
            summary["legacy"][key]["selected"] = len(items)
            summary["legacy"][key]["finished"] = len(fin)
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--family", choices=["4b", "lite"], default="4b")
    ap.add_argument("--tag")
    ap.add_argument("--suites", default="all")
    ap.add_argument("--limit", type=int, default=160, help="items per suite (tools_fixed: limit // 40 per kind); 0 = all")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-shared", type=int, default=8, help="prefix groups with at least this many live items share the prefix KV")
    ap.add_argument("--kv-slots", type=int, default=2)
    ap.add_argument("--batch-tokens", type=int, default=160000, help="padded tokens per batch (prompt + new), bounds KV and prefill memory")
    ap.add_argument("--chunk", type=int, default=600, help="agent conversations per pooled chunk (summary rewritten after each)")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: no batch starts with less than 3 min left; finished work is written, exit 0")
    ap.add_argument("--max-new", type=int, default=MAX_NEW_AGENT, help="tokens per agent round (512; a reply reaching it is hit_max)")
    ap.add_argument("--max-new-chat", type=int, default=640, help="tokens per chat-mode reply (eval_hermit3: 640; maths chat 400)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", default=None, help="results dir: rescore the saved transcripts, no model")
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(0 if dry_run(a.family, a.base, a.limit, a.bs, a.min_shared) else 1)
    if a.rescore:
        rescore(a.rescore, a.family)
        return
    if not a.base or not a.tag:
        ap.error("--base and --tag are required (or --dry-run / --rescore)")
    outdir = os.path.join(EV5, "results", a.tag)
    os.makedirs(outdir, exist_ok=True)
    sp = os.path.join(outdir, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {"tag": a.tag}
    summary.update({"family": a.family, "base": a.base, "adapter": a.adapter, "limit": a.limit, "bs": a.bs})
    summary.setdefault("suites", {}); summary.setdefault("legacy", {}); summary.setdefault("timing", {})
    if time_left(a.deadline) < 180:
        print(f"[{a.tag}] deadline already reached, nothing run"); json.dump(summary, open(sp, "w"), indent=1); return

    def save():
        json.dump(summary, open(sp, "w"), indent=1)

    want = expand_suites(a.suites, a.family)
    sel = selected_items(want, a.family, a.limit)
    store = Store(outdir)
    done, rows_cache = {}, {}
    for name, items in sel.items():
        for it in items:
            d = store.rows.get((name, it["id"]))
            if d is not None:
                done[(name, it["id"])] = Finished(it, a.family, d)
    if done:
        print(f"[{a.tag}] {len(done)} conversations already finished (transcripts.jsonl), skipped")
    pending = []
    for name in sorted(sel, key=priority_key):
        pending += [Conv(it, a.family, MAX_ROUNDS.get(name, 6)) for it in sel[name] if (name, it["id"]) not in done]
    tail = [c for c in pending if c.suite == "cloud_examples_v3"]
    pending = [c for c in pending if c.suite != "cloud_examples_v3"]
    eng = Engine(a.base, a.adapter, a.family, a.threads, a.kv_slots)
    t_all = time.time()

    def finished(c):
        store.add(c)
        done[(c.suite, c.item["id"])] = c

    def run_chunks(convs):
        for j in range(0, len(convs), a.chunk):
            if time_left(a.deadline) < 180:
                return False
            chunk = convs[j:j + a.chunk]
            ts = time.time()
            print(f"[{a.tag}] chunk {j // a.chunk + 1}: {len(chunk)} conversations, suites {dict(Counter(c.suite for c in chunk))}", flush=True)
            complete = run_pool(eng, chunk, a.bs, a.min_shared, finished, max_new=a.max_new, log=lambda x: print(x, flush=True),
                                deadline=a.deadline, budget=a.batch_tokens)
            summary["timing"][f"chunk_{len(summary['timing'])}"] = round(time.time() - ts)
            summary["engine"] = dict(eng.stats)
            write_results(outdir, summary, sel, done, a.family, rows_cache)
            print(S.format_summary({n: summary["suites"][n] for n in dict.fromkeys(c.suite for c in chunk) if n in summary["suites"]}), flush=True)
            if not complete:
                return False
        return True

    ok = run_chunks(pending)
    # ---- chat-mode suites (no tools), then cloud_examples_v3
    for n in want:
        if not ok or time_left(a.deadline) < 180:
            ok = False
            break
        ts = time.time()
        if n in ("legacy_math_v2", "legacy_math_legacy"):
            key = n.replace("legacy_", "") + "_chat"
            if key in summary["legacy"]:
                continue
            rows = [json.loads(l) for l in open(os.path.join(EV5, n + ".jsonl"))]
            rows = [r for r in rows if a.family != "lite" or "lite" in (r.get("family") or ["4b", "lite"])]
            rows = rows[:a.limit] if a.limit else rows
            sm, det = run_math_chat(eng, rows, a.bs, min(400, a.max_new_chat))
            summary["legacy"][key] = sm
            json.dump(det, open(os.path.join(outdir, f"{n}_chat.json"), "w"), indent=1, ensure_ascii=False)
        elif n == "legacy_multiturn" and "multiturn" not in summary["legacy"]:
            scripts = [json.loads(l) for l in open(os.path.join(EV5, "legacy_multiturn.jsonl"))]
            r = run_multiturn(eng, scripts, a.bs, max_new=a.max_new_chat)
            json.dump(r, open(os.path.join(outdir, "legacy_multiturn_detail.json"), "w"), indent=1, ensure_ascii=False)
            nt = sum(x["turns"] for x in r)
            summary["legacy"]["multiturn"] = {"convs": len(r), "replies": nt, "rep_within_replies": sum(x["rep_within"] for x in r),
                                              "rep_across_replies": sum(x["rep_across"] for x in r), "para_repeats": sum(x["para_repeat"] for x in r),
                                              "empty": sum(x["empty"] for x in r), "runaway": sum(x["runaway"] for x in r),
                                              "convs_clean": sum(1 for x in r if not (x["rep_within"] or x["rep_across"] or x["para_repeat"] or x["empty"] or x["runaway"])),
                                              "mean_words": sum(sum(x["words"]) for x in r) / max(nt, 1)}
        elif n == "legacy_chat" and "chat" not in summary["legacy"]:
            rows = [json.loads(l) for l in open(os.path.join(EV5, "legacy_chat.jsonl"))]
            r = run_chat(eng, rows, a.bs, max_new=a.max_new_chat)
            json.dump(r, open(os.path.join(outdir, "legacy_chat_detail.json"), "w"), indent=1, ensure_ascii=False)

            def agg(kind):
                x = [y for y in r if y["kind"] == kind]
                return {"n": len(x), "refused": sum(y["refused"] for y in x), "middle_dot": sum(y["middle_dot"] for y in x), "boiler": sum(y["boiler"] for y in x),
                        "empty": sum(y["empty"] for y in x), "hit_max": sum(y["hit_max"] for y in x), "mean_words": sum(y["words"] for y in x) / max(len(x), 1)}
            summary["legacy"]["chat"] = {k: agg(k) for k in ("tone", "hardline", "legit_edge")}
        elif n == "bots" and "bots" not in summary["legacy"]:
            r = run_bots(eng, a.bs, max_new=a.max_new_chat)
            json.dump(r, open(os.path.join(outdir, "bots_detail.json"), "w"), indent=1)
            summary["legacy"]["bots"] = {"pass": sum(x["ok"] for x in r), "n": len(r)}
        else:
            continue
        summary["timing"][n] = round(time.time() - ts)
        save()
        print(f"[{a.tag}] {n} done in {time.time() - ts:.0f}s, gen tokens so far {eng.ntok}", flush=True)
    if ok and tail:
        ok = run_chunks(tail)
    summary["timing"]["total"] = round(time.time() - t_all) + summary["timing"].get("total_before", 0)
    summary["gen_tokens"] = eng.ntok
    summary["engine"] = dict(eng.stats)
    summary["complete"] = ok and all(summary["suites"].get(n, {}).get("finished") == len(sel[n]) for n in sel if n in S.AGENT_SUITES)
    write_results(outdir, summary, sel, done, a.family, rows_cache)
    print(S.format_summary(summary["suites"]))
    print(json.dumps(summary["legacy"], indent=1)[:3000])
    print(f"[{a.tag}] {'complete' if summary['complete'] else 'PARTIAL (deadline)'} in {time.time() - t_all:.0f}s. Gates: python gen4/make_results4.py {a.tag} --family {a.family}")


if __name__ == "__main__":
    main()
