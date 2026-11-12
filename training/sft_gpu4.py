#!/usr/bin/env python3
"""LoRA SFT for Hermit 1.3 / 1.3 Lite on sft5 (tools v3, tool packs). From pkg3 train/sft_gpu3.py; what changed:

  * rendering: gen4/fmt4.render(messages, pack, family). 4B rows use meta.pack, Lite rows meta.pack_lite (data5/STATS.md).
    --family 4b trains on every row; --family lite only on rows whose meta.family_ok contains "lite" (the others are skipped
    and counted in the log). Every encoded row's length is compared with the token count assemble5 measured
    (meta.tokens_4b / meta.tokens_lite); --render-check N proves on N random rows that the encoded ids decode to exactly
    fmt4.render_text, that tokenizing the whole text gives the same ids, and that the trained tokens are exactly the
    assistant segments.
  * steps: every optimizer step holds exactly --seqs-per-step rows (the last step of an epoch may hold fewer). The plan
    depends only on the rows, the seed and --step-grouping, never on the micro-batch limits: the A100 (big micro-batches)
    and the A6000 (small ones) train on the same steps, and an OOM resume with smaller micro-batches continues the SAME plan
    at the same step. pkg3 packed micro-batches until a step had 16 or more rows (16 to 25) and rebuilt a different plan on
    resume.
  * prefix KV: the trick of pkg3 (system prompt + tool block encoded once under no_grad, gradients through everything
    after it), now per pack. The first rendered segment (system prompt + the pack's tool block, device word, name /
    hermit.md suffixes) is the cache key. --step-grouping key (default) fills each step from one key where it can, so most
    steps need one prefix forward (pkg3-style mixing needs about 6 on sft5); leftovers of every key are mixed into the
    remaining steps. The step order is shuffled, so packs alternate from step to step. A step's micro-batches are grouped
    by key and each prefix KV is freed as soon as its key's micro-batches are done.
  * micro-batches: padded suffix tokens <= --max-micro-tokens, rows <= --max-micro-seqs and, when --max-micro-ctx is set,
    rows x (prefix + longest suffix) <= --max-micro-ctx (the KV copies and the GQA-repeated K/V saved for backward grow
    with the context, not with the suffix).
  * memory probe: before training, the heaviest micro-batches of the plan (--probe N) run forward+backward once without an
    update. An OOM, or a peak above --mem-target-gb, shrinks the micro-batch limits by a quarter and probes again (6 tries);
    the limits that pass are logged and used. run_remote4.sh still resumes after a mid-run OOM with smaller limits.
  * --max-len defaults to the family caps of data5/STATS.md (4B 8192, Lite 6144: longest rows 7,641 / 4,713 tokens, so no
    row is dropped); rows over it are counted and logged, never silently lost.
  * --plan-only: encode, build the plan, print token/prefix/padding statistics as JSON and exit (no model load).
  * the val loss pass walks the val rows key by key (one prefix KV alive at a time).
Unchanged from sft_gpu3: assistant-only loss, logits only at trained positions, LoRA on attn+MLP (fp32 LoRA weights on a
bf16 base), AdamW, warmup then cosine to 10%, gradient clip 1.0, checkpoints `last` every --save-every steps (adapter +
optimizer + scheduler + step), `step<N>` at every val pass, `final` at the end.

Data: data5/sft5_{train,val}.jsonl rows {"messages": [...], "tools": bool, "meta": {"pack", "pack_lite", "family_ok", ...}}.
"""
import argparse, json, math, os, random, sys, time
from collections import Counter

import torch
import torch.nn.functional as F

DEV = torch.device(os.environ.get('HERMIT_DEVICE', 'cuda'))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gen4"))
import fmt4 as H   # 30-tool registry, packs, SYSTEM_AGENT_v3, device word (gen4/fmt4.py)

MAX_LEN = {"4b": 8192, "lite": 6144}       # data5/STATS.md caps (longest rows 7,641 and 4,713 tokens)
META_TOK = {"4b": "tokens_4b", "lite": "tokens_lite"}


def log(msg, fh=None):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def row_pack(ex, family):
    """The pack this row is rendered with for `family`, or None when the row is not trained for that family."""
    m = ex.get("meta") or {}
    if family == "lite":
        if "lite" not in (m.get("family_ok") or []):
            return None
        return m.get("pack_lite") or []
    return m.get("pack") or []


def encode(tok, ex, family, max_len):
    """-> dict(key, p, ids, lab, meta_tok) or a reason string ("family", "too_long", "no_target")."""
    pack = row_pack(ex, family)
    if pack is None:
        return "family"
    segs = H.render(ex["messages"], pack, family)
    assert not segs[0][1], "the first segment (the prefix) must not be trainable"
    prefix_text = segs[0][0]
    p_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    ids, lab = [], []
    for text, train in segs[1:]:
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        lab += t if train else [-100] * len(t)
    if len(p_ids) + len(ids) > max_len:
        return "too_long"
    if all(l == -100 for l in lab):
        return "no_target"
    return {"key": prefix_text, "p": p_ids, "ids": ids, "lab": lab, "meta_tok": (ex.get("meta") or {}).get(META_TOK[family])}


def encode_all(tok, rows, family, max_len):
    items, why = [], Counter()
    for r in rows:
        e = encode(tok, r, family, max_len)
        if isinstance(e, str):
            why[e] += 1
        else:
            items.append(e)
    return items, why


def make_steps(items, seqs_per_step, seed, grouping="key"):
    """-> list of optimizer steps, each a list of exactly `seqs_per_step` rows (the last one may be short).

    The plan depends only on the rows, the seed and the grouping, never on the micro-batch limits.
    grouping "key" (default): rows are grouped by prefix key (pack + device word + notes); each key's rows are shuffled and
      cut into full steps, so most steps hold one prefix and need one prefix forward; the leftovers of every key (and all
      single-row prefixes) are shuffled together, kept next to rows of the same key, and cut into mixed steps. The step
      order is shuffled, so packs alternate from step to step; only the rows inside one step share a pack.
    grouping "mixed": rows shuffled over all keys (pkg3-like mixing; about 6 prefix forwards per step on sft5)."""
    rng = random.Random(seed)
    if grouping == "mixed":
        rows = list(items); rng.shuffle(rows)
        return [rows[i:i + seqs_per_step] for i in range(0, len(rows), seqs_per_step)]
    by = {}
    for it in items:
        by.setdefault(it["key"], []).append(it)
    steps, rest = [], []
    for key in sorted(by, key=lambda k: (len(by[k]), k)):
        lst = list(by[key])
        rng.shuffle(lst)
        full = len(lst) // seqs_per_step * seqs_per_step
        steps += [lst[i:i + seqs_per_step] for i in range(0, full, seqs_per_step)]
        rest += lst[full:]
    rng.shuffle(rest)
    rest.sort(key=lambda x: x["key"])          # stable: leftovers of one key stay together, so a mixed step needs few prefixes
    steps += [rest[i:i + seqs_per_step] for i in range(0, len(rest), seqs_per_step)]
    rng.shuffle(steps)
    return steps


def micro_batches(step, max_micro_tokens, max_micro_seqs, max_micro_ctx=0):
    """Split one step into micro-batches: one prefix key per micro-batch (keys in sorted order, so a key's micro-batches
    are consecutive), rows sorted by suffix length; a micro-batch keeps padded suffix tokens (longest x rows) under
    max_micro_tokens, at most max_micro_seqs rows and, if max_micro_ctx > 0, rows x (prefix + longest suffix) under
    max_micro_ctx. A single row over a limit always gets its own micro-batch."""
    by = {}
    for it in step:
        by.setdefault(it["key"], []).append(it)
    micros = []
    for key in sorted(by):
        cur = []
        for it in sorted(by[key], key=lambda x: len(x["ids"])):
            L = max([len(x["ids"]) for x in cur] + [len(it["ids"])])
            n = len(cur) + 1
            if cur and (L * n > max_micro_tokens or n > max_micro_seqs or (max_micro_ctx and n * (len(it["p"]) + L) > max_micro_ctx)):
                micros.append(cur); cur = []
            cur.append(it)
        if cur:
            micros.append(cur)
    return micros


def micro_shape(m):
    """(rows, prefix tokens, padded suffix length) of a micro-batch."""
    return len(m), len(m[0]["p"]), max(len(x["ids"]) for x in m)


def plan_stats(plan, limits):
    """Counts the cost model and the log need: prefix forwards (one per distinct prefix in a step) and their tokens,
    micro-batches, padded suffix tokens, attention work (sum over padded suffix tokens of their context), trained tokens."""
    s = Counter()
    for st in plan:
        keys = {}
        for it in st:
            keys.setdefault(it["key"], len(it["p"]))
        s["prefix_forwards"] += len(keys)
        s["prefix_tokens_forwarded"] += sum(keys.values())
        s["prefix_attn"] += sum(p * p / 2 for p in keys.values())
        for m in micro_batches(st, *limits):
            b, P, S = micro_shape(m)
            s["micro_batches"] += 1
            s["padded_suffix_tokens"] += b * S
            s["suffix_attn"] += b * S * (P + S / 2)
            s["kv_ctx_tokens"] += b * (P + S)
        s["suffix_tokens"] += sum(len(x["ids"]) for x in st)
        s["trained_tokens"] += sum(sum(1 for l in x["lab"] if l != -100) for x in st)
        s["rows_processed"] += len(st)
    return dict(s)


def plan_sha(plan):
    """Fingerprint of the step plan (which rows, in which steps, in which order): equal across micro-batch limits."""
    import hashlib
    h = hashlib.sha256()
    for st in plan:
        for it in st:
            h.update(hashlib.sha1((it["key"] + "\x00" + ",".join(map(str, it["ids"]))).encode()).digest())
        h.update(b"|")
    return h.hexdigest()[:16]


def prefix_kv(model, p_ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([p_ids], device=DEV), use_cache=True)
    return out.past_key_values


def expand_cache(cache, b, model):
    from transformers import DynamicCache
    new = DynamicCache(config=model.config)
    for li, layer in enumerate(cache.layers):
        new.update(layer.keys.expand(b, -1, -1, -1).contiguous(), layer.values.expand(b, -1, -1, -1).contiguous(), li)
    return new


def micro_loss(model, base, micro, kv, pad_id):
    b = len(micro)
    P = len(micro[0]["p"])
    S = max(len(x["ids"]) for x in micro)
    ids = torch.full((b, S), pad_id, dtype=torch.long)
    lab = torch.full((b, S), -100, dtype=torch.long)
    att = torch.zeros((b, P + S), dtype=torch.long)
    att[:, :P] = 1
    for i, x in enumerate(micro):
        n = len(x["ids"])
        ids[i, :n] = torch.tensor(x["ids"]); lab[i, :n] = torch.tensor(x["lab"]); att[i, P:P + n] = 1
    pos = torch.arange(P, P + S).unsqueeze(0).expand(b, -1)
    ids, lab, att, pos = ids.to(DEV), lab.to(DEV), att.to(DEV), pos.to(DEV)
    cache = expand_cache(kv, b, base)
    # PEFT wraps: model.base_model.model is the CausalLM; its .model is the decoder
    dec = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    hs = dec(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
    # predict token t+1 from position t; the first suffix token is predicted from the prefix's last position,
    # but that label is never trained (it's '<|im_start|>' of a user turn), so we only shift within the suffix.
    tgt = lab[:, 1:]
    h = hs[:, :-1]
    m = tgt != -100
    logits = lm_head(h[m]).float()
    loss_sum = F.cross_entropy(logits, tgt[m], reduction="sum")
    return loss_sum, int(m.sum())


def render_check(tok, rows, family, n, seed, fh):
    """n random rows of the family: (1) the encoded ids decode to exactly fmt4.render_text, (2) tokenizing that whole
    text gives the same ids (segment-wise tokenization == what the app's tokenizer sees), (3) the trained tokens decode
    to exactly the trainable segments (assistant replies + <|im_end|>), (4) the prefix is the first segment and is never
    trained, (5) the length equals the count assemble5 stored in meta. Returns True when every row passes."""
    cand = [r for r in rows if row_pack(r, family) is not None]
    pick = random.Random(seed).sample(cand, min(n, len(cand)))
    bad = 0
    for r in pick:
        pack = row_pack(r, family)
        e = encode(tok, r, family, 10 ** 9)
        segs = H.render(r["messages"], pack, family)
        text = H.render_text(r["messages"], pack, family)
        full = e["p"] + e["ids"]
        probs = []
        if tok.decode(full, skip_special_tokens=False) != text:
            probs.append("decode != fmt4.render_text")
        if tok(text, add_special_tokens=False)["input_ids"] != full:
            probs.append("whole-text tokenization differs")
        if tok.decode([i for i, l in zip(e["ids"], e["lab"]) if l != -100], skip_special_tokens=False) != "".join(t for t, tr in segs if tr):
            probs.append("trained tokens != trainable segments")
        if e["key"] != segs[0][0] or tok.decode(e["p"], skip_special_tokens=False) != segs[0][0]:
            probs.append("prefix != first segment")
        if e["meta_tok"] is not None and e["meta_tok"] != len(full):
            probs.append(f"length {len(full)} != meta {e['meta_tok']}")
        src = (r.get("meta") or {}).get("source")
        name = (r.get("meta") or {}).get("pack_lite_name" if family == "lite" else "pack_name")
        log(f"render-check {family} {src:18} pack {name:16} tokens {len(full):5} trained {sum(1 for l in e['lab'] if l != -100):4} "
            + ("OK" if not probs else "FAIL " + "; ".join(probs)), fh)
        bad += bool(probs)
    log(f"render-check {family}: {len(pick) - bad}/{len(pick)} rows identical to fmt4.render_text "
        f"(decode, whole-text tokenization, trained segments, prefix, assemble5 length)", fh)
    return bad == 0


def heaviest(plan, limits, k):
    """Up to 2k distinct micro-batches of the plan that need the most memory: the top k by padded suffix tokens
    (activations) and the top k by rows x context (KV copies, GQA-repeated K/V)."""
    ms = [m for st in plan for m in micro_batches(st, *limits)]
    pick, seen = [], set()
    for f in (lambda m: (micro_shape(m)[0] * micro_shape(m)[2], micro_shape(m)[0] * sum(micro_shape(m)[1:])),
              lambda m: (micro_shape(m)[0] * sum(micro_shape(m)[1:]), micro_shape(m)[0] * micro_shape(m)[2])):
        for m in sorted(ms, key=f, reverse=True)[:k]:
            if id(m) not in seen:
                seen.add(id(m)); pick.append(m)
    return pick


def probe(model, micros, pad_id, params):
    """Forward+backward of each micro-batch, no update. -> peak allocated GB (0 on CPU). Raises OOM."""
    if DEV.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for m in micros:
        model.eval(); kv = prefix_kv(model, m[0]["p"]); model.train()
        l, k = micro_loss(model, model, m, kv, pad_id)
        (l / max(k, 1)).backward()
        del l, kv
        for p in params:
            p.grad = None
    return torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--family", choices=["4b", "lite"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seqs-per-step", type=int, default=16)
    ap.add_argument("--max-micro-tokens", type=int, default=3072)
    ap.add_argument("--max-micro-seqs", type=int, default=16, help="cap on rows per micro-batch (the prefix KV is copied per row)")
    ap.add_argument("--max-micro-ctx", type=int, default=0, help="cap on rows x (prefix + longest suffix) per micro-batch; 0 = off")
    ap.add_argument("--max-len", type=int, default=0, help="prefix+suffix token cap; 0 = the family cap of data5/STATS.md (4B 8192, Lite 6144)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0, help="smoke runs: use only the first N training rows of this family")
    ap.add_argument("--plan-only", action="store_true", help="encode + step plan statistics as JSON, then exit (no model)")
    ap.add_argument("--render-check", type=int, default=0, help="check N random rows against fmt4.render_text, then exit (no model)")
    ap.add_argument("--step-grouping", choices=["key", "mixed"], default="key", help="key: rows of one prefix share a step (default); mixed: pkg3-like")
    ap.add_argument("--probe", type=int, default=3, help="memory probe: forward+backward of the 2N heaviest micro-batches before training; 0 = off")
    ap.add_argument("--mem-target-gb", type=float, default=0.0, help="shrink the micro-batch limits until the probe peak is under this; 0 = only on OOM")
    a = ap.parse_args()
    a.max_len = a.max_len or MAX_LEN[a.family]

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, "train.log"), "a")
    log(f"args {vars(a)}", fh)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    tok = AutoTokenizer.from_pretrained(a.base)
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")

    rows = [json.loads(l) for l in open(a.train)]
    if a.render_check:
        ok = render_check(tok, rows, a.family, a.render_check, a.seed + 7, fh)
        sys.exit(0 if ok else 1)
    if a.limit_rows:
        rows = [r for r in rows if row_pack(r, a.family) is not None][: a.limit_rows]
    items, why = encode_all(tok, rows, a.family, a.max_len)
    keys = Counter(i["key"] for i in items)
    mism = sum(1 for i in items if i["meta_tok"] is not None and i["meta_tok"] != len(i["p"]) + len(i["ids"]))
    log(f"train rows {len(rows)} -> encoded {len(items)} for family {a.family} (skipped: not {a.family} {why['family']}, over --max-len {a.max_len}: "
        f"{why['too_long']}, no trained token {why['no_target']}); prefixes {len(keys)} (single-row {sum(1 for v in keys.values() if v == 1)}); "
        f"longest row {max(len(i['p']) + len(i['ids']) for i in items)} tokens; suffix tokens {sum(len(i['ids']) for i in items)}; "
        f"trained tokens {sum(sum(1 for l in i['lab'] if l != -100) for i in items)}; prefix tokens (once per row) {sum(len(i['p']) for i in items)}; "
        f"length != assemble5 meta: {mism}", fh)
    if why["too_long"]:
        log(f"WARNING {why['too_long']} rows dropped over --max-len {a.max_len}", fh)
    if mism:
        log(f"WARNING {mism} rows encode to a different length than assemble5 measured (meta {META_TOK[a.family]})", fh)
    val = []
    if a.val:
        vr = [json.loads(l) for l in open(a.val)]
        val, vwhy = encode_all(tok, vr, a.family, a.max_len)
        log(f"val rows {len(vr)} -> encoded {len(val)} ({dict(vwhy)})", fh)

    steps_per_epoch = make_steps(items, a.seqs_per_step, a.seed, a.step_grouping)
    n_ep = math.ceil(a.epochs)
    plan = []
    for ep in range(n_ep):
        s = make_steps(items, a.seqs_per_step, a.seed + ep, a.step_grouping) if ep else steps_per_epoch
        if ep == n_ep - 1 and a.epochs != n_ep:
            s = s[: int(len(s) * (a.epochs - ep))]
        plan += s
    full_sha = plan_sha(plan)                    # before --max-steps: the same for every run of these rows, seed and grouping
    if a.max_steps:
        plan = plan[: a.max_steps]
    total = len(plan)
    limits = [a.max_micro_tokens, a.max_micro_seqs, a.max_micro_ctx]
    ps = plan_stats(plan, limits)
    log(f"optimizer steps {total} ({len(steps_per_epoch)} per epoch, grouping {a.step_grouping}, rows per step {a.seqs_per_step}, plan {full_sha}); "
        f"micro-batches {ps['micro_batches']} (limits tokens {limits[0]}, rows {limits[1]}, ctx {limits[2] or 'off'}); prefix forwards "
        f"{ps['prefix_forwards']} ({ps['prefix_forwards'] / max(total, 1):.2f} per step), prefix tokens forwarded {ps['prefix_tokens_forwarded']}; "
        f"padded suffix tokens {ps['padded_suffix_tokens']} (real {ps['suffix_tokens']})", fh)
    if a.plan_only:
        print(json.dumps({"family": a.family, "rows": len(rows), "encoded": len(items), "skipped": dict(why), "prefixes": len(keys),
                          "single_row_prefixes": sum(1 for v in keys.values() if v == 1), "steps": total, "steps_per_epoch": len(steps_per_epoch),
                          "longest": max(len(i["p"]) + len(i["ids"]) for i in items), "meta_len_mismatch": mism,
                          "suffix_tokens_epoch": sum(len(i["ids"]) for i in items), "trained_tokens_epoch": sum(sum(1 for l in i["lab"] if l != -100) for i in items),
                          "limits": limits} | ps))
        return

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(DEV)
    torch.backends.cuda.matmul.allow_tf32 = True
    model.config.use_cache = True
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    log(f"trainable params {sum(p.numel() for p in params):,}", fh)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99), fused=(DEV.type == 'cuda'))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))

    start = 0
    ck = os.path.join(a.out, "last")
    if os.path.exists(os.path.join(ck, "state.pt")):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(ck, "adapter_model.safetensors"), device=str(DEV))
        set_peft_model_state_dict(model, sd)
        for p in params:
            p.data = p.data.float()
        st = torch.load(os.path.join(ck, "state.pt"), weights_only=False, map_location=DEV)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]
        log(f"resumed from step {start} of the same plan", fh)

    # memory probe: the heaviest micro-batches of what is left of the plan, before any update
    if a.probe and start < total:
        for attempt in range(6):
            micros = heaviest(plan[start:], limits, a.probe)
            shapes = [micro_shape(m) for m in micros]
            try:
                peak = probe(model, micros, pad_id, params)
                over = a.mem_target_gb and peak > a.mem_target_gb
                log(f"memory probe {attempt + 1}: limits tokens {limits[0]} rows {limits[1]} ctx {limits[2] or 'off'}; {len(micros)} heaviest micro-batches "
                    f"(rows, prefix, suffix) {shapes}; peak {peak:.1f} GB" + (f" > target {a.mem_target_gb} GB" if over else ""), fh)
            except torch.cuda.OutOfMemoryError:
                over = True
                log(f"memory probe {attempt + 1}: OOM at limits tokens {limits[0]} rows {limits[1]} ctx {limits[2] or 'off'} {shapes}", fh)
            for p in params:
                p.grad = None
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
            if not over:
                break
            limits = [max(256, limits[0] * 3 // 4), max(1, limits[1] * 3 // 4), limits[2] * 3 // 4 if limits[2] else 0]
        else:
            log("memory probe: no limits passed in 6 tries; training with the smallest (an OOM restarts from `last`)", fh)
        ps2 = plan_stats(plan[start:], limits)
        log(f"training with limits tokens {limits[0]} rows {limits[1]} ctx {limits[2] or 'off'}: {ps2['micro_batches']} micro-batches for "
            f"{total - start} steps", fh)

    def save(step, where):
        d = os.path.join(a.out, where)
        tmp = d + ".tmp"
        model.save_pretrained(tmp)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, os.path.join(tmp, "state.pt"))
        if os.path.exists(d):
            import shutil; shutil.rmtree(d)
        os.rename(tmp, d)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            cur, kv = None, None
            for it in sorted(val[:96], key=lambda x: x["key"]):     # one prefix KV alive at a time
                if it["key"] != cur:
                    kv = None
                    kv, cur = prefix_kv(model, it["p"]), it["key"]
                l, k = micro_loss(model, model, [it], kv, pad_id)
                tot += float(l); n += k
            kv = None
        model.train()
        return tot / max(n, 1)

    model.train()
    t0 = time.time(); ema = None
    for step in range(start, total):
        ts = time.time()
        micros = micro_batches(plan[step], *limits)
        ntrain = sum(sum(1 for l in x["lab"] if l != -100) for m in micros for x in m)
        loss_acc = 0.0
        cur, kv, npfx = None, None, 0
        for m in micros:                         # micro-batches of one key are consecutive: one prefix KV alive at a time
            if m[0]["key"] != cur:
                kv = None
                model.eval()                     # no dropout for the cached prefix
                kv, cur = prefix_kv(model, m[0]["p"]), m[0]["key"]
                model.train()
                npfx += 1
            l, k = micro_loss(model, model, m, kv, pad_id)
            (l / ntrain).backward()
            loss_acc += float(l.detach())
        kv = None
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        stoks = sum(len(x["ids"]) for m in micros for x in m)
        dt = time.time() - ts
        lv = loss_acc / ntrain
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        el = time.time() - t0
        eta = el / (step + 1 - start) * (total - step - 1)
        mem = (f"{torch.cuda.max_memory_allocated() / 1e9:.1f}G/{torch.cuda.max_memory_reserved() / 1e9:.1f}G" if DEV.type == 'cuda' else "0G")
        log(f"step {step + 1}/{total} loss {lv:.4f} ema {ema:.4f} lr {sched.get_last_lr()[0]:.2e} seqs {sum(len(m) for m in micros)} "
            f"micro {len(micros)} pfx {npfx} tok {stoks} {stoks / dt:.0f} tok/s step {dt:.1f}s eta {eta / 3600:.2f}h mem {mem}", fh)
        if (step + 1) % a.save_every == 0 or step + 1 == total:
            save(step + 1, "last")
        if val and ((step + 1) % a.eval_every == 0 or step + 1 == total):
            log(f"val loss {evaluate():.4f} at step {step + 1}", fh)
            save(step + 1, f"step{step + 1}")
    save(total, "final")
    log("done", fh)


if __name__ == "__main__":
    main()
