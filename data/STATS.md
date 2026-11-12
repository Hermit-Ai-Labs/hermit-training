# data5 / sft5: statistics

Written by `gen4/assemble5.py` (Qwen3 tokenizers: Qwen3-4B-Instruct-2507 for 4B, Qwen3-1.7B for Lite). Render: `fmt4.render(messages, meta.pack, "4b")`; Lite: `fmt4.render(messages, meta.pack_lite, "lite")` on rows with `"lite"` in `meta.family_ok`.

**train 21,361 rows, val 414 rows.** Unique training rows 20,253 + 1,108 restraint top-up copies.

## Rows per source

| source | in | train | val | lite-ok (train) |
|---|---:|---:|---:|---:|
| ask_wording | 570 | 536 | 11 | 536 |
| auto_pages | 480 | 470 | 10 | 470 |
| contacts_pay | 520 | 504 | 10 | 504 |
| drift_fix | 300 | 289 | 6 | 261 |
| followups_stale | 750 | 733 | 15 | 563 |
| lite_cloud | 260 | 244 | 5 | 189 |
| multistep | 700 | 686 | 14 | 292 |
| notes_name | 500 | 488 | 10 | 468 |
| orders_protect | 420 | 409 | 8 | 266 |
| portfolio_caps | 500 | 486 | 10 | 483 |
| recovery_errors | 600 | 577 | 12 | 561 |
| rounding_brief | 750 | 727 | 15 | 479 |
| safety_mixed | 200 | 195 | 5 | 195 |
| spot_solana | 520 | 509 | 10 | 509 |
| aave |  | 548 | 11 | 548 |
| activity_tool |  | 240 | 5 | 240 |
| buy_jupiter |  | 329 | 7 | 329 |
| close_position |  | 329 | 6 | 320 |
| core_math |  | 3,434 | 70 | 3,434 |
| core_reads |  | 1,614 | 28 | 1,169 |
| core_trade |  | 1,475 | 30 | 1,221 |
| file_qa |  | 490 | 10 | 391 |
| honesty |  | 483 | 10 | 483 |
| knowledge_fix |  | 312 | 6 | 312 |
| multisend_file |  | 415 | 8 | 335 |
| multisend_typed |  | 441 | 9 | 437 |
| open_app_tool |  | 294 | 5 | 294 |
| restraint_code |  | 89 | 3 | 89 |
| restraint_misc |  | 141 | 5 | 141 |
| restraint_strategy |  | 261 | 2 | 261 |
| restraint_teacher |  | 1,749 | 19 | 1,748 |
| restraint_voice |  | 43 | 0 | 43 |
| route_rounding |  | 490 | 10 | 490 |
| send_nochain |  | 588 | 12 | 588 |
| stocks_price |  | 155 | 5 | 155 |
| trade_view |  | 588 | 12 | 113 |
| **total** | | **21,361** | **414** | **18,917** |

data4 rows in train: 6,853 (32.1%); sft4 rows: 14,508.

## sft4 rows dropped or rewritten as now wrong (PLAN_v3 0.7 + refusal search)

sft4 rows in: 14,442 train + 290 val.

| reason | rows |
|---|---:|
| dropped: kind neg_cancel_order | 14 |
| dropped: kind neg_sell | 12 |
| dropped: nav_imperative | 12 |
| dropped: no_automation | 5 |
| dropped: plan_0.7 | 2 |
| dropped: kind neg_nav (no page match) | 1 |
| **dropped total** | **46** |
| rewritten to open_page: neg_nav | 17 |
| rewritten to open_page: not_activity_howto | 7 |

Refusal-search drops by source/kind: restraint_teacher/multiturn nav_imperative x7, restraint_teacher/answer nav_imperative x5, restraint_teacher/multiturn no_automation x3, restraint_teacher/answer no_automation x2, restraint_teacher/answer plan_0.7 x2.

## Leakage and de-duplication

Leak index: 4,283 eval items (eval2, eval3, eval4 and eval5/safety_mixed; contexts and scripts).

- leakage drops: 74 ask_wording/sequence 20, recovery_errors/sequence 8, core_reads/exact 8, core_reads/sequence 6, portfolio_caps/sequence 3, rounding_brief/exact 3, lite_cloud/sequence 3, lite_cloud/jaccard 3, orders_protect/exact 2, notes_name/sequence 2, recovery_errors/exact 2, followups_stale/exact 2, rounding_brief/sequence 2, ask_wording/exact 2, activity_tool/exact 2, lite_cloud/exact 1, drift_fix/sequence 1, drift_fix/exact 1, aave/sequence 1, buy_jupiter/sequence 1
- duplicate user texts dropped: 282 core_reads (same as core_reads) 243, core_reads (same as drift_fix) 6, core_reads (same as recovery_errors) 6, core_reads (same as ask_wording) 4, core_trade (same as core_trade) 4, rounding_brief (same as recovery_errors) 2, spot_solana (same as spot_solana) 1, orders_protect (same as orders_protect) 1, portfolio_caps (same as spot_solana) 1, recovery_errors (same as portfolio_caps) 1, rounding_brief (same as notes_name) 1, ask_wording (same as ask_wording) 1, drift_fix (same as rounding_brief) 1, drift_fix (same as portfolio_caps) 1, knowledge_fix (same as ask_wording) 1, core_reads (same as spot_solana) 1, activity_tool (same as drift_fix) 1, activity_tool (same as core_reads) 1, core_reads (same as auto_pages) 1, activity_tool (same as ask_wording) 1
- sft4's deliberate repeats collapsed: 722 restraint_strategy 237, restraint_teacher 205, restraint_code 152, restraint_voice 125, restraint_misc 3
- over the 4B cap (8192): 0
- failing gen4/check4.py (PLAN_v3 C9), dropped: 11 contacts_pay: arg_omission (open_page#tab=payees not asked for) 6; lite_cloud: arg_omission (open_page#tab=positions not asked for) 2; lite_cloud: arg_omission (open_page#tab=payees not asked for) 1; lite_cloud: c6b (search_tokens tag 'Mint on' not repeated) 1; drift_fix: arg_omission (open_page#tab=explore not asked for) 1

## Packs (4B, `meta.pack`)

| pack | tools | train | val |
|---|---:|---:|---:|
| core | 12 | 6,157 (28.8%) | 125 |
| core+trade | 20 | 4,897 (22.9%) | 80 |
| all | 30 | 3,564 (16.7%) | 72 |
| core+solana | 16 | 1,852 (8.7%) | 36 |
| core+pay | 16 | 1,730 (8.1%) | 38 |
| chat | 0 | 1,476 (6.9%) | 32 |
| core+earn | 16 | 1,459 (6.8%) | 25 |
| core+earn+pay | 20 | 96 (0.4%) | 2 |
| core+solana+pay | 20 | 67 (0.3%) | 3 |
| core+trade_lite | 16 | 42 (0.2%) | 1 |
| core+solana+earn | 19 | 21 (0.1%) | 0 |

Where the pack came from (train+val): router 11,208, generator 4,827, all30 2,149, router_fallback 986. `router_fallback` = the row calls a tool outside the router's choice and gets the smallest router pack holding its calls: 986 rows (trade_view->core+trade 326, close_position->core+trade 136, buy_jupiter->core+solana 111, core_trade->core+trade 90, core_reads->core+earn 37, rounding_brief->core+trade 35, recovery_errors->core+solana 31, aave->core+earn 31, multisend_typed->core+pay 30, rounding_brief->core+earn 28, core_reads->core+trade 23, multisend_file->core+pay 16).

Generator packs replaced (not a pack the app can send, or calls outside it): spot_solana (23 tools, not a router pack) 53.

Packs per source (train):

| source | all | chat | core | core+earn | core+earn+pay | core+pay | core+solana | core+solana+earn | core+solana+pay | core+trade | core+trade_lite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ask_wording | 536 |  |  |  |  |  |  |  |  |  |  |
| auto_pages |  |  | 172 |  |  | 298 |  |  |  |  |  |
| contacts_pay |  |  | 168 |  |  | 336 |  |  |  |  |  |
| drift_fix | 42 |  | 72 | 38 |  |  | 1 | 1 |  | 135 |  |
| followups_stale | 733 |  |  |  |  |  |  |  |  |  |  |
| lite_cloud |  |  | 77 | 35 |  | 34 | 56 |  |  |  | 42 |
| multistep |  |  | 257 | 233 |  | 12 | 69 |  |  | 115 |  |
| notes_name | 72 |  | 171 | 35 |  |  | 41 |  |  | 169 |  |
| orders_protect | 21 |  |  |  |  |  | 20 |  |  | 368 |  |
| portfolio_caps |  |  | 272 | 28 | 2 | 19 | 65 |  |  | 100 |  |
| recovery_errors | 76 |  | 259 | 64 | 1 | 26 | 66 |  |  | 85 |  |
| rounding_brief | 94 |  | 314 | 94 | 21 | 26 | 27 | 1 |  | 150 |  |
| safety_mixed |  |  | 131 |  |  | 14 | 50 |  |  |  |  |
| spot_solana | 6 |  | 29 |  |  |  | 462 |  |  | 12 |  |
| aave | 70 |  | 5 | 466 |  |  | 4 | 3 |  |  |  |
| activity_tool | 39 |  | 182 | 15 |  |  | 4 |  |  |  |  |
| buy_jupiter | 52 |  | 48 |  |  |  | 229 |  |  |  |  |
| close_position | 56 |  | 2 |  |  |  |  |  |  | 271 |  |
| core_math | 350 | 1128 | 917 | 92 | 65 | 19 | 263 |  |  | 600 |  |
| core_reads | 230 | 79 | 649 | 175 |  |  | 43 | 3 |  | 435 |  |
| core_trade | 206 |  |  |  |  |  |  |  |  | 1269 |  |
| file_qa | 80 |  | 175 |  | 2 | 218 |  |  | 1 | 14 |  |
| honesty | 79 |  | 240 | 49 |  | 29 | 40 |  |  | 46 |  |
| knowledge_fix | 50 |  | 187 | 32 |  |  | 14 | 2 |  | 27 |  |
| multisend_file | 55 |  |  |  |  | 360 |  |  |  |  |  |
| multisend_typed | 68 |  | 18 |  |  | 289 |  |  | 66 |  |  |
| open_app_tool | 56 |  | 178 | 24 |  |  | 24 | 1 |  | 11 |  |
| restraint_code | 17 | 7 | 37 |  |  |  | 14 |  |  | 14 |  |
| restraint_misc | 37 |  | 35 |  |  |  | 69 |  |  |  |  |
| restraint_strategy | 43 | 39 | 139 | 4 |  |  |  |  |  | 36 |  |
| restraint_teacher | 215 | 217 | 587 | 69 | 5 | 50 | 162 | 10 |  | 434 |  |
| restraint_voice | 4 | 6 | 20 |  |  |  | 6 |  |  | 7 |  |
| route_rounding | 65 |  | 378 | 6 |  |  | 15 |  |  | 26 |  |
| send_nochain | 95 |  | 410 |  |  |  | 83 |  |  |  |  |
| stocks_price | 24 |  | 28 |  |  |  | 25 |  |  | 78 |  |
| trade_view | 93 |  |  |  |  |  |  |  |  | 495 |  |

## family_ok (PLAN_v3 C2)

- train: 4B only 2,444, 4B + Lite 18,917; val: 4B only 47, 4B + Lite 367.
- not Lite because: more than 3 steps 1,382, calls fit no Lite pack 985, conversation over 2k tokens 124 (train+val).
- Lite packs (`meta.pack_lite`, train): core 7,600, core+trade_lite 4,311, core+solana 2,069, core+pay 1,822, core+earn 1,640, chat 1,475.
- data4 rows vs the generator's own family_ok: same 2,932, generator 4b -> 4b+lite 1,369, generator 4b+lite -> 4b 369.

## Restraint share (tools present, no tool call)

sft4 (v2) 0.1663; sft5 before top-up 0.1207; after top-up 0.1663 (1108 copies: restraint_teacher 845, restraint_strategy 123, restraint_misc 76, restraint_code 44, restraint_voice 20); val 0.1304.

## Token lengths (with the pack prefix)

- Hermit 4B, all train rows (21,361): mean 3,065, p50 2,917, p90 4,986, p99 5,496, max 7,641. Cap 8192.
- Hermit Lite, lite-ok train rows (18,917): mean 2,418, p50 2,599, p90 3,019, p99 4,096, max 4,713. Cap 6144.
- Total tokens per epoch (train, prefix included): 4B 65,478,791; Lite 45,741,856.
- System + tool block prefix (Face ID, no name/notes): 4b all 4,774, 4b chat 144, 4b core 1,962, 4b core+earn 2,443, 4b core+earn+pay 3,117, 4b core+pay 2,636, 4b core+solana 2,586, 4b core+solana+earn 2,949, 4b core+solana+pay 3,260, 4b core+trade 3,231, 4b core+trade_lite 2,616, lite chat 144, lite core 1,962, lite core+earn 2,443, lite core+pay 2,636, lite core+solana 2,586, lite core+trade_lite 2,616.
- Longest 4B rows: 7,641 (multisend_file/plain, all); 7,634 (multisend_file/plain, all); 7,575 (file_qa/cut, all).
- Longest Lite rows: 4,713 (multisend_file/plain, core+pay); 4,678 (file_qa/cut, core+pay); 4,673 (file_qa/exp_sums, core+pay).

Device word (train): Face ID 14,605, Touch ID 6,756.

Validation: `gen4/check4.py` (PLAN_v3 C9) runs over every row of both files; see its output for the result.

