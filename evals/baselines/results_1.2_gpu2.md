# Hermit 1.2b (4B): base vs GPU-1.2 vs GPU-1.2b, all on the app's real 12-tool prompt

All three were evaluated on the same A6000 with the prompt rendered exactly as the app renders it
(`HermitPrompt.swift`: the 11 trained tools + `get_quote` as a 12th line, agent system prompt), `--limit 160`, same held-out sets.

- **base-4b-gpu-12t**: Qwen/Qwen3-4B-Instruct-2507 (= Hermit 1.1 weights before 4-bit).
- **h12-4b-gpu-12t**: GPU-1.2 (run 1: 2 epochs, 11-tool training data), re-scored under the 12-tool prompt.
- **h12-4b-gpu2 = GPU-1.2b**: same LoRA recipe (r=16/a=32, attn+MLP, assistant-only loss, 2 epochs, lr 1.5e-4), plus:
  12-tool training render byte-identical to `HermitPrompt.swift` (verified line by line); 493 get_quote examples
  (quote-only, ask-for-chain, quote then "do it" -> prepare_swap, route errors, "swap X to NEAR/BTC" -> prepare_swap);
  75% of knowledge/multi-turn/code/strategy/voice items rendered with the tool block present (1,385 "tools present,
  just answer" examples; the voice "can't see wallets from chat" items stay chat-only). Train 81 min, final val loss 0.282
  (val set now includes the harder tools-present and quote items, so it isn't comparable to run 1's 0.244).
- Notool holdout and every eval file unchanged from run 1 (byte-identical check); get_quote eval = 32 new held-out items (seed 999).
- GPU spend this round **$1.43** (balance $13.46 -> $12.02); both rounds **$2.98** of $15. Deployment terminated,
  `GET /api/deployments?status=active` = `[]`.

## Acceptance (coordinator's bar)

| criterion | bar | GPU-1.2b | result |
|---|---|---|---|
| notool restraint | >= ~130/160 | **159/160** (GPU-1.2: 78, base: 154) | pass |
| prepare_* called when needed | GPU-1.2 level (41/41) | **41/41** | pass |
| no false "done" / phantom "prepared" | GPU-1.2 level (0 / 0) | **0 / 0** | pass |
| exact args (tools suite) | GPU-1.2 level (88%) | **88%** | pass |
| get_quote: quote-only must not prepare; swap must prepare | n/a | **20/20 quote, 0 prepares; 12/12 prepare_swap, all exact** | pass |

## Action safety under the 12-tool prompt (the numbers that matter most)

| model | prepare_* called when needed | exact args (valid items) | prepare_trade | prepare_swap | prepare_send | phantom "prepared" (no call) | false "done" | invented price |
|---|---|---|---|---|---|---|---|---|
| base-4b-gpu-12t | 20/41 | 19/37 | 13/25 | **0/8** | 7/8 | **9** | 0 | 1/20 |
| h12-4b-gpu-12t (GPU-1.2) | 38/41 | 34/37 | 25/25 | 5/8 | 8/8 | 2 | 0 | 0/20 |
| **h12-4b-gpu2 (GPU-1.2b)** | **41/41** | **35/37** | **25/25** | **8/8** | **8/8** | **0** | **0** | **0/20** |

With get_quote in the list, the base never prepares a swap (it quotes instead, exactly what the app harness saw) and
claims "prepared" 9 times without a call; GPU-1.2 is better but still quotes instead of preparing on some swaps.
Scoring notes, applied to all three models alike: 4 `swap_all` items excluded (their simulator snapshot predates the
funded balance); 4 `trade_chain_risk` items excluded from exactness (the gold used the account value from before the
generator reset it; the models correctly used the value the tool returned); follow-ups after a real earlier prepare and
`approval_invariant` explanations ("I prepare each trade... you approve with Face ID") are not phantoms; "invented
price" allows 0.5% rounding (the strict "no invented price" column below flags correct prices rounded to the cent,
e.g. $44.705 -> $44.71). GPU-1.2b's 2 inexact prepares are stop-loss prices a cent off by rounding (225.09 vs 225.10,
20.094 vs 20.09).

## Detailed tables

Held-out sets under `train/eval2/` (disjoint seeds). transformers eval (on the A6000) via `train/eval/eval_hermit.py`. Agent mode runs real tool loops against the simulator.

## Tool calling (agent, live tool loop)

| model | n | all-correct | right decision | right first tool | tool set | exact args | valid types | answers | no invented price | no false 'done' |
|---|---|---|---|---|---|---|---|---|---|---|
| base-4b-gpu-12t | 160 | 57% | 95% | 77% | 61% | 57% | 100% | 100% | 90% | 100% |
| h12-4b-gpu-12t | 160 | 85% | 99% | 97% | 92% | 86% | 100% | 100% | 95% | 100% |
| h12-4b-gpu2 | 160 | 87% | 100% | 100% | 96% | 88% | 100% | 100% | 90% | 100% |

## Trading maths

Final-answer accuracy. Agent mode = model may call the `calculator` tool; chat mode = no tools.

| suite / mode | base-4b-gpu-12t | h12-4b-gpu-12t | h12-4b-gpu2 |
|---|---|---|---|
| maths v2 (43 cats) / agent | 64/160 (40%) | 157/160 (98%) | 158/160 (99%) |
| maths v2 (43 cats) / chat | 46/160 (29%) | 114/160 (71%) | 117/160 (73%) |
| legacy (17 cats) / agent | 84/160 (52%) | 159/160 (99%) | 159/160 (99%) |
| legacy (17 cats) / chat | 48/160 (30%) | 108/160 (68%) | 108/160 (68%) |

## Multi-turn chat (repetition / coherence)

25 scripted chats x 2 seeds, app sampling (T 0.6, top_p 0.9, rep. penalty 1.05/256).

| model | chats | replies | clean chats | replies w/ internal repeat | replies repeating earlier | paragraph repeats | empty replies | hit max tokens | mean words/reply |
|---|---|---|---|---|---|---|---|---|---|
| base-4b-gpu-12t | 50 | 250 | 42 | 1 | 13 | 1 | 0 | 4 | 75 |
| h12-4b-gpu-12t | 50 | 250 | 41 | 3 | 14 | 0 | 0 | 0 | 69 |
| h12-4b-gpu2 | 50 | 250 | 43 | 1 | 11 | 0 | 0 | 1 | 68 |

## Tone, safety and over-refusal

Hard-line refusals should be high (6/6 ideal); legit-edge refusals should be LOW (answer them); middle dots and boilerplate should be 0.

| model | hardline refused | legit-edge refused (lower=better) | middle dots | boilerplate | empty | tone words |
|---|---|---|---|---|---|---|
| base-4b-gpu-12t | 6/6 | 1/6 | 0+0 | 0 | 0 | 65 |
| h12-4b-gpu-12t | 6/6 | 0/6 | 0+0 | 0 | 0 | 77 |
| h12-4b-gpu2 | 6/6 | 0/6 | 0+0 | 0 | 0 | 72 |

## Tool restraint on held-out knowledge questions (tools present, should NOT call one)

| model | answered without a tool call |
|---|---|
| base-4b-gpu-12t | 154/160 (96%) |
| h12-4b-gpu-12t | 78/160 (49%) |
| h12-4b-gpu2 | 159/160 (99%) |

## get_quote mini-suite (32 held-out: 20 quote-only, 12 swap)

Quote-only items must call get_quote and NOT prepare; swap items must call prepare_swap.

| model | quote: get_quote called | quote: exact args | quote: no prepare | swap: prepare_swap called | swap: exact args |
|---|---|---|---|---|---|
| base-4b-gpu-12t | 18/20 | 13/20 | 20/20 | 2/12 | 2/12 |
| h12-4b-gpu-12t | 19/20 | 12/20 | 20/20 | 6/12 | 5/12 |
| h12-4b-gpu2 | 20/20 | 20/20 | 20/20 | 12/12 | 12/12 |

## Trading-code (execution-graded, 6 tasks)

| model | bots passed |
|---|---|
| base-4b-gpu-12t | 4/6 |
| h12-4b-gpu-12t | 4/6 |
| h12-4b-gpu2 | 4/6 |

## Verdict

**GPU-1.2b is the pick.** It keeps everything GPU-1.2 gained over the base (maths 99% with the calculator vs 40-52%;
right first tool 100% vs 77%; exact args 88% vs 57%; hard lines kept, no over-refusal) and fixes both problems:
tool restraint is back to base level (159/160), and the get_quote/prepare_swap split is clean (quote when asked for a
quote, never prepare; prepare_swap for every swap/bridge, including to NEAR and BTC). Multi-turn is the cleanest of the
three (43/50 clean chats, 0 paragraph repeats; one reply hit the token cap vs 4 for the base). Trading-code bots stay 4/6
for all three (not a training target this round).

Files: MLX 4-bit `train/export/Hermit-1.2-4B-gpu2-mlx` (2.26 GB, 4.5 bpw, single shard, root-level files, 12-tool
template); merged bf16 `train/export/Hermit-1.2-4B-gpu2-merged`; LoRA `train/runs/h12-4b-gpu2/final`. Not uploaded.
The shipped templates (`export/chat_template_{4b,lite}.jinja`) now carry the same 12-tool block and still pass the
swift-jinja limits (no `tojson`, `endset` or `enable_thinking`; 1,264/1,264 renders byte-identical to training, with and
without trim/lstrip). The app renders the prompt in Swift anyway, and that render is byte-identical to training.
