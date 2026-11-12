# Hermit 1.2 (4B): base vs GPU-1.2 (vs CPU-1.2 when its eval lands)

- **base-4b-gpu** = Qwen/Qwen3-4B-Instruct-2507 (= shipped Hermit 1.1 weights, before 4-bit), shipped Hermit 1.2 template + tools.
- **h12-4b-gpu** = LoRA r=16/a=32 (attn+MLP), assistant-only loss, same data and recipe as the CPU run, **2 epochs** (1048 steps,
  lr 1.5e-4 cosine) on 1x RTX A6000 48 GB (Spheron / massed-compute, dedicated). Train 67 min; final val loss 0.244
  (CPU run, 1 epoch: 0.279 at step 300/510).
- Same held-out eval sets and harness (`eval/eval_hermit.py`, `--limit 160`), greedy decoding with rep. penalty 1.05/256;
  multi-turn uses the app's sampling. Both models evaluated on the same GPU, so they're directly comparable.
- **CPU-1.2 (h12-4b, 1 epoch)**: still finishing on this server (training at step ~500/510, then a multi-hour CPU eval).
  Its column will be added when `results12/h12-4b/summary.json` exists. The CPU pipeline's *base* eval crashed on
  2026-11-07 (simulator didn't catch a ValueError when the base passed `"unknown"` as a number; fixed), so the
  GPU base above is the baseline.
- GPU spend: **$1.54** (balance $15.00 -> $13.46), deployment terminated, `GET /api/deployments?status=active` = `[]`.

## The numbers that matter most: action safety (prepare_* actually called, no phantom/false "done")

The real-app harness showed Hermit 1.1 saying "Trade prepared" without calling `prepare_trade` and "Sent" for a send that
was only prepared. Same failure reproduced here on the base. Rescored by hand-checked rules (see notes):

| model | prepare_* called when needed | ...with exact args | prepare_trade | prepare_swap | prepare_send | phantom "prepared" (no call) | false "done" |
|---|---|---|---|---|---|---|---|
| base-4b-gpu | 26/41 (63%) | 23/41 (56%) | 13/25 | 5/8 | 8/8 | **6** | 0 |
| **h12-4b-gpu** | **41/41 (100%)** | **37/41 (90%)** | **25/25** | **8/8** | **8/8** | **0** | **0** |

Base phantom examples: "Liquidation price is $168.70 ... Trade is prepared for review. Approve with Face ID" with no
`prepare_trade` call (trade_chain_liq x3, trade_chain_risk x2, trade_chain_stop x1). GPU-1.2 calls price -> position_math
-> prepare_trade on every chain item. Neither model called a prepare_* tool when none was asked for (0/115 for 1.2).
Notes: 4 `swap_all` items are excluded from both rows: an eval-set bug (their simulator snapshot predates the balance
the generator added, so "nothing to swap" is the correct answer and both models were scored wrong). Follow-up questions
like "did it go through?" after a real earlier prepare, and "I've opened Add funds" (a screen, not money) are not
counted as phantom/false-done; the raw regex counts were 3 and 1 for 1.2, all of them these cases.


## Detailed tables

Held-out sets under `train/eval2/` (disjoint seeds). transformers eval (on the A6000) via `train/eval/eval_hermit.py`. Agent mode runs real tool loops against the simulator.

## Tool calling (agent, live tool loop)

| model | n | all-correct | right decision | right first tool | tool set | exact args | valid types | answers | no invented price | no false 'done' |
|---|---|---|---|---|---|---|---|---|---|---|
| base-4b-gpu | 160 | 59% | 94% | 83% | 66% | 59% | 99% | 100% | 90% | 100% |
| h12-4b-gpu | 160 | 87% | 99% | 99% | 94% | 88% | 100% | 100% | 95% | 100% |

## Trading maths

Final-answer accuracy. Agent mode = model may call the `calculator` tool; chat mode = no tools.

| suite / mode | base-4b-gpu | h12-4b-gpu |
|---|---|---|
| maths v2 (43 cats) / agent | 65/160 (41%) | 156/160 (98%) |
| maths v2 (43 cats) / chat | 46/160 (29%) | 114/160 (71%) |
| legacy (17 cats) / agent | 82/160 (51%) | 159/160 (99%) |
| legacy (17 cats) / chat | 48/160 (30%) | 108/160 (68%) |

## Multi-turn chat (repetition / coherence)

25 scripted chats x 2 seeds, app sampling (T 0.6, top_p 0.9, rep. penalty 1.05/256).

| model | chats | replies | clean chats | replies w/ internal repeat | replies repeating earlier | paragraph repeats | empty replies | hit max tokens | mean words/reply |
|---|---|---|---|---|---|---|---|---|---|
| base-4b-gpu | 50 | 250 | 42 | 1 | 13 | 1 | 0 | 4 | 75 |
| h12-4b-gpu | 50 | 250 | 41 | 3 | 14 | 0 | 0 | 0 | 69 |

## Tone, safety and over-refusal

Hard-line refusals should be high (6/6 ideal); legit-edge refusals should be LOW (answer them); middle dots and boilerplate should be 0.

| model | hardline refused | legit-edge refused (lower=better) | middle dots | boilerplate | empty | tone words |
|---|---|---|---|---|---|---|
| base-4b-gpu | 6/6 | 1/6 | 0+0 | 0 | 0 | 65 |
| h12-4b-gpu | 6/6 | 0/6 | 0+0 | 0 | 0 | 77 |

## Tool restraint on held-out knowledge questions (tools present, should NOT call one)

| model | answered without a tool call |
|---|---|
| base-4b-gpu | 154/160 (96%) |
| h12-4b-gpu | 74/160 (46%) |

## Trading-code (execution-graded, 6 tasks)

| model | bots passed |
|---|---|
| base-4b-gpu | 4/6 |
| h12-4b-gpu | 4/6 |

## Verdict

**GPU-1.2 wins clearly and is the candidate for the real-app check.** Versus the base it:
- calls the prepare tool every time an action is needed (41/41 vs 26/41), with exact arguments 90% vs 56%, and never
  claims a trade/send is prepared or done without the call (0 vs 6 phantom);
- routes maths through the calculator every time (160/160 calls) and gets **98-99%** of held-out trading maths right
  in agent mode (base 41-51%); without tools, worked-step maths 68-71% (base 29-30%);
- picks the right tool first 99% (base 83%), exact args 88% overall (base 59%), invents fewer prices (95% vs 90% clean);
- keeps the hard lines (6/6 refused), refuses none of the legitimate security/aggressive questions (base 1/6), no middle
  dots or boilerplate, no runaway generations in multi-turn (base 4), repetition on par with base.

**One regression to fix before shipping: tool restraint.** On held-out knowledge questions asked with the tool block
present, 1.2 answers directly only 74/160 (base 154/160). The extra calls are all read-only (calculator 29, get_price 20,
search_markets 12, hyperliquid_account 9, get_yields 6, get_balances 3, position_math 2; **no prepare_* calls**), so
nothing unsafe happens, but it adds latency and some calls are pointless (e.g. calculator on a symbolic formula).
Cause: `assemble2.py` rendered all teacher Q&A without the tool block, so the model never saw "tools present, just answer".
Fix (cheap): re-render ~half the teacher Q&A + multi-turn with tools present and retrain (another ~$1 on the A6000).

## Before the app can use 1.2
- The shipped template emits a **fixed** tool block (no `tojson` in swift-jinja). It contains the 11 trained tools; the
  app's new 12th tool **get_quote is not in it, so the model can't see it**. Add get_quote to `hermit_tools.json`,
  regenerate the template (`export/make_templates.py`), and add get_quote examples ("how much NEAR for 20 USDC" ->
  quote, answer, no prepare) in the retrain above. (Not done in this run: training had started when the finding arrived.)
- Use `SYSTEM_AGENT` (gen2/hermit_fmt.py) as the system prompt when tools are passed; stop on `<|im_end|>`.
- MLX 4-bit of GPU-1.2: `train/export/Hermit-1.2-4B-gpu-mlx` (2.26 GB, 4.5 bpw, single shard, root-level files).
  LoRA adapter: `train/runs/h12-4b-gpu/final`; merged bf16: `train/export/Hermit-1.2-4B-gpu-merged`. **Not uploaded.**
