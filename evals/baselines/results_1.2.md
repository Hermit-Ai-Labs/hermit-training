# Hermit 1.2 evaluation: base vs fine-tuned

Held-out sets under `train/eval2/` (disjoint seeds). CPU transformers eval via `train/eval/eval_hermit.py`. Agent mode runs real tool loops against the simulator.

## Tool calling (agent, live tool loop)

| model | n | all-correct | right decision | right first tool | tool set | exact args | valid types | answers | no invented price | no false 'done' |
|---|---|---|---|---|---|---|---|---|---|---|
| base-4b | 160 | 57% | 96% | 78% | 62% | 57% | 100% | 100% | 90% | 100% |
| h12-4b | 160 | 86% | 100% | 100% | 95% | 88% | 100% | 100% | 95% | 100% |

## Trading maths

Final-answer accuracy. Agent mode = model may call the `calculator` tool; chat mode = no tools.

| suite / mode | base-4b | h12-4b |
|---|---|---|
| maths v2 (43 cats) / agent | 64/160 (40%) | 158/160 (99%) |
| maths v2 (43 cats) / chat | 52/160 (32%) | 110/160 (69%) |
| legacy (17 cats) / agent | 82/160 (51%) | 159/160 (99%) |
| legacy (17 cats) / chat | 50/160 (31%) | 106/160 (66%) |

## Multi-turn chat (repetition / coherence)

25 scripted chats x 2 seeds, app sampling (T 0.6, top_p 0.9, rep. penalty 1.05/256).

| model | chats | replies | clean chats | replies w/ internal repeat | replies repeating earlier | paragraph repeats | empty replies | hit max tokens | mean words/reply |
|---|---|---|---|---|---|---|---|---|---|
| base-4b | 50 | 250 | 40 | 2 | 16 | 3 | 0 | 2 | 71 |
| h12-4b | 50 | 250 | 44 | 2 | 9 | 4 | 0 | 0 | 62 |

## Tone, safety and over-refusal

Hard-line refusals should be high (6/6 ideal); legit-edge refusals should be LOW (answer them); middle dots and boilerplate should be 0.

| model | hardline refused | legit-edge refused (lower=better) | middle dots | boilerplate | empty | tone words |
|---|---|---|---|---|---|---|
| base-4b | 6/6 | 1/6 | 0+0 | 0 | 0 | 66 |
| h12-4b | 6/6 | 0/6 | 0+0 | 0 | 0 | 64 |

## Tool restraint on held-out knowledge questions (tools present, should NOT call one)

| model | answered without a tool call |
|---|---|
| base-4b | 153/160 (96%) |
| h12-4b | 70/160 (44%) |

## Trading-code (execution-graded, 6 tasks)

| model | bots passed |
|---|---|
| base-4b | 4/6 |
| h12-4b | 4/6 |
