# Data

| File | What it is |
|---|---|
| `sft5_train.jsonl.gz` | 21,361 training conversations, gzipped (about 80 MB unpacked) |
| `sft5_val.jsonl` | 414 validation conversations |
| `sft5_report.json`, `STATS.md` | What the assembler dropped, rewrote and kept, plus token statistics. Written by `generators/gen4/assemble5.py` |
| `sample_conversations.json` | 20 conversations, pretty-printed, one per behaviour. The `meta` field is shortened here |
| `LICENSE` | CC BY 4.0 |

Read them with `python tools/load_data.py` (prints `train 21361` and `val 414`), or `zcat data/sft5_train.jsonl.gz | head -n 1`.

## Row format

One JSON object per line:

```
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "", "tool_calls": [{"name": "get_price", "arguments": {...}}]}, {"role": "tool", "content": "{...}"}, {"role": "assistant", ...}],
 "tools": true,
 "meta": {"source": "...", "kind": "...", "bio": "Face ID", "pack": [...tool names...], "pack_lite": [...], "family_ok": ["4b", "lite"], "origin": "...", ...}}
```

- `tools` is true when the prompt carries a tool block. 1,476 training rows are plain chat with no tools.
- `meta.pack` is the list of tools the 4B prompt carries for this row. `meta.pack_lite` is the same for Lite. An empty list means plain chat.
- `meta.family_ok` says which model trains on the row: `["4b"]` only, or `["4b", "lite"]`. 18,917 training rows are usable by Lite (at most 3 tool calls, under 2,048 tokens after the prompt, calls fit a Lite pack).
- `meta.bio` is the device word in the prompt, Face ID (iPhone) or Touch ID (Mac).
- `meta.origin` names the pool file and line the row came from.
- To get the exact text the model saw, render the row with `generators/gen4/fmt4.py`: `fmt4.render(row["messages"], row["meta"]["pack"], "4b")` (Lite: `row["meta"]["pack_lite"]`, `"lite"`). The loss is on the assistant segments only.

## What is in it

| Source | Train | Val | Lite-ok (train) | Made by | What it teaches |
|---|---:|---:|---:|---|---|
| `spot_solana` | 509 | 10 | 509 | gen4 | Solana token buys and sells by symbol or mint address, token search, look-alike symbols, refusals for thin or frozen tokens |
| `orders_protect` | 409 | 8 | 266 | gen4 | Stop loss and take profit, cancelling orders, reading perps positions |
| `contacts_pay` | 504 | 10 | 504 | gen4 | Saved contacts, paying by name, scheduled payments worded as reminders |
| `auto_pages` | 470 | 10 | 470 | gen4 | Automations, opening app pages, pointing to the right page when no tool exists |
| `portfolio_caps` | 486 | 10 | 483 | gen4 | Portfolio and positions reads, what the app can and cannot do, app facts |
| `multistep` | 686 | 14 | 292 | gen4 | Chains of several tool calls across old and new tools |
| `notes_name` | 488 | 10 | 468 | gen4 | Following the user's own notes (hermit.md) and the name they asked to be called |
| `recovery_errors` | 577 | 12 | 561 | gen4 | What to do after a tool returns an error: retry once, say what happened, do not invent data |
| `followups_stale` | 733 | 15 | 563 | gen4 | Follow-up turns ("do the same for ETH"), carried state, not reusing stale data |
| `rounding_brief` | 727 | 15 | 479 | gen4 | Short answers with rounded numbers |
| `ask_wording` | 536 | 11 | 536 | gen4 | When to ask a question and when to act; iPhone and Mac wording; several languages |
| `lite_cloud` | 244 | 5 | 189 | gen4 | Longer trajectories for the Lite prompt, copying addresses and amounts exactly |
| `drift_fix` | 289 | 6 | 261 | gen4 | Older tools whose results changed shape in v3 |
| `safety_mixed` | 195 | 5 | 195 | gen4 | Wallet-safety requests mixed with a safe ask: decline the unsafe half in one line, do the rest |
| `aave` | 548 | 11 | 548 | gen3 | Aave v3 deposit and withdraw |
| `activity_tool` | 240 | 5 | 240 | gen3 | Reading recent activity and status of sends, swaps and buys |
| `buy_jupiter` | 329 | 7 | 329 | gen3 | Buying Solana tokens through Jupiter versus swapping |
| `close_position` | 329 | 6 | 320 | gen3 | Closing perps positions, partial closes, ambiguous coins |
| `file_qa` | 490 | 10 | 391 | gen3 | Questions about attached CSV, XLSX and PDF text |
| `honesty` | 483 | 10 | 483 | gen3 | Never say money moved without a result; check activity for status |
| `knowledge_fix` | 312 | 6 | 312 | gen3 | Correct app and crypto facts (recovery phrase advice, allowances, impermanent loss) |
| `multisend_file` | 415 | 8 | 335 | gen3 | prepare_multisend from an attached sheet |
| `multisend_typed` | 441 | 9 | 437 | gen3 | prepare_multisend from a typed list |
| `open_app_tool` | 294 | 5 | 294 | gen3 | Opening a dapp by name or URL, refusing look-alike or unsafe URL schemes |
| `route_rounding` | 490 | 10 | 490 | gen3 | Quote versus prepare, route wording (NEAR Intents), rounding |
| `send_nochain` | 588 | 12 | 588 | gen3 | Sending when the user names no network: leave the network out, let the app choose |
| `stocks_price` | 155 | 5 | 155 | gen3 | Stock and index prices |
| `trade_view` | 588 | 12 | 113 | gen3 | Trade views and follow-up questions |
| `core_math` | 3,434 | 70 | 3,434 | gen2 | Trading maths with the calculator tool (43 categories, answers computed in code) |
| `core_reads` | 1,614 | 28 | 1,169 | gen2 | Price, balance, position and yield reads |
| `core_trade` | 1,475 | 30 | 1,221 | gen2 | Perps trade preparation with stops, liquidation price and risk sizing |
| `restraint_teacher` | 1,749 | 19 | 1,748 | gen2 | Knowledge answers and multi-turn chats with the tools present and no call needed (written by an open teacher model, filtered) |
| `restraint_strategy` | 261 | 2 | 261 | gen | Strategy questions, hand-written answers |
| `restraint_code` | 89 | 3 | 89 | gen | Trading-code requests; every reference solution was run |
| `restraint_misc` | 141 | 5 | 141 | gen2 | Send-scam warnings (126) and quote questions with no tool (15) |
| `restraint_voice` | 43 | 0 | 43 | hand-written | Short voice, honest-limits and refusal examples (43) |
| **total** | **21,361** | **414** | **18,917** | | |

Counts match `STATS.md`. Of the 21,361 training rows, 20,253 are distinct conversations. The other 1,108 are identical copies of tool-restraint examples added back on purpose, so the share of rows that show "tools present, no call needed" stays at the level of the earlier data (16.6%). The validation file has no copies.

By tool, the training set holds 27,512 tool calls over all 30 tools. The most used are `calculator` (6,818), `get_price` (3,494), `prepare_trade` (2,425), `get_balances` (1,766) and `prepare_send` (1,709). The least used are `open_add_funds` (36), `add_contact` (64) and `search_markets` (91).

Tool packs in the training set (4B): `core` 28.8%, `core+trade` 22.9%, all 30 tools 16.7%, `core+solana` 8.7%, `core+pay` 8.1%, plain chat 6.9%, `core+earn` 6.8%, and a few small combinations.

## Lineage

The training file is the fifth version of the Hermit data. Each version kept what still held and added or replaced the rest.

```
gen/ + gen2/   ->  sft3  (Hermit 1.2b set: calculator, reads, trade prep, teacher answers, strategy, code, voice)
                    |
sft3 + gen3/   ->  sft4  (adds 14 pools for the 19-tool app: Aave, multisend, honesty, file questions ...)
                    |
sft4 + gen4/   ->  sft5  (adds 14 pools for the 30-tool app and tool packs)   <- this folder
```

`generators/gen4/assemble5.py` produced sft5 from the sft4 files and the gen4 pools (docstring at the top lists every step). What it did to the older rows:

- Dropped 46 sft4 rows that were wrong for tools v3 (for example "I can only prepare buys" when `prepare_sell` exists).
- Rewrote 24 navigation answers ("tap Earn in the tab bar") into an `open_page` call and a "button below" answer.
- Replaced the system prompt of every tool row with `spec/SYSTEM_AGENT_v3.txt` (the device word kept).
- Dropped 74 training rows that matched a user turn of any test item (4,283 test items checked), 282 more with the same user text as an earlier row, and 11 that failed `gen4/check4.py`.
- Chose a tool pack for every row: the generator's own pack when the app can send it, otherwise 15% all 30 tools and 85% the pack the app's router would pick, widened if the row calls a tool outside it.

The assembler needs the sft4 files and the gen4 pools, which are not in this repo (see the top-level README).

## Where the text comes from

- **Generated from templates and a simulator** (gen2 agent pools, gen3 and gen4): the user's request is a template filled with random slots, the tool results come from the simulator, and the assistant's answer is a template filled only from those results and the user's text. Numbers that need maths come from code (`calculator` results and the answer agree because both are computed).
- **Written by an open model** (`restraint_teacher`, 1,768 rows including validation): Qwen3-30B-A3B-Instruct-2507 (Apache-2.0) wrote user questions and Hermit-style answers from a facts sheet (`generators/gen2/app_facts.txt`). The rows were filtered for length, finish, banned characters, claimed live prices, repeated sentences and wrong arithmetic. No output of a closed API is in the data.
- **Hand-written** (`restraint_strategy`, `restraint_code`, `restraint_voice`): 393 rows. Strategy answers are written by hand; every code answer was run against its checker.

## Addresses, keys and names

- **Addresses are random.** EVM addresses are 40 random hex digits from a seeded generator, then given an EIP-55 checksum (`generators/gen2/hermit_fmt.py`, `evm_addr`). Solana addresses are 32 random bytes in base58 (`sol_addr`). Across the whole set there are 8,294 distinct EVM addresses in 8,387 rows; only USDC's public contract appears more than 3 times.
- **Real public identifiers are used as examples where a real one is needed**: the USDC contract, popular Solana token mints (for example WIF and JUP), and ENS names like `vitalik.eth` and `satoshi.eth` as send targets.
- **No private keys and no recovery phrases.** There is no run of 11 or more words from the BIP-39 list anywhere. The 64-digit hex strings that do appear are fake transaction hashes in sample spreadsheets and NEAR account ids built from random bytes.
- **Contacts and emails are made up**: first names from short lists with invented surnames, and addresses like `ben@acme.io` or `zoe@mail.com` on example domains.
- **No secrets.** `api_key`, `Bearer` and `password` appear only inside answers that explain how to call an API or why a note should not hold a phrase.

## License

CC BY 4.0, see `LICENSE`. The base models and the teacher model are Apache-2.0, see `../NOTICE.md`.

## Sample names

A few sample first names and one internal metadata label differ in this published copy from the files on the training machine. Nothing else differs: the same 21,361 and 414 conversations, the same tool calls, amounts and addresses.
