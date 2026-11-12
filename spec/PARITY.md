# Render parity: lab (gen3/fmt3 + gen2) vs app (HermitPrompt.swift), checked 2026-11-11

Script: `spec/parity_check.py` (read only on both sides; reruns the whole check). Checked with the 19 tools of `spec/hermit_tools_v2.json`,
`SYSTEM_AGENT_v2.txt` with {BIO}=Face ID, one conversation with a parallel call pair, two tool results, an assistant text turn,
a non-ASCII user turn and the generation prompt. The app text was rebuilt by transcribing `agentSystem`, `trainedToolLines`,
`extraToolLines`, `toolsBlock` and `render` (parsed out of the Swift file, not retyped).

## Result
- System prompt: `agentSystem` == SYSTEM_AGENT_v2 (Face ID), byte for byte. The two "price / view" sentences are already in the app.
- Tool lines: the app's 15 lines (11 trained + 4 extra) are byte-identical to the first 15 lines of the 19-tool registry, same order.
  That includes `prepare_swap` ("through NEAR Intents": already in the app, nothing to change) and `prepare_send` (`chain` still in `required`, as the
  app has it; `fmt3.validate_call` accepts a call without it).
- Full render (system + tools block + user/assistant/tool framing + generation prompt): lab render == app render once the 4 lines below are added.
  With today's 15 lines the only differences are those 4 missing lines (`prepare_buy`, `prepare_close`, `get_activity`, `open_app`).
- Lab: `fmt3.render_text` == gen2 `hermit_fmt.render_text` pointed at the v2 registry. No lab-side mismatch found, no gen3 change needed.
- Training rows (`data3/sft4_train.jsonl`, 14442): every tools row starts with SYSTEM_AGENT_v2 (row's device word) plus only the name / hermit.md suffix;
  no Touch ID row has a leftover "Face ID" in later messages; no row contains the text `neutralize` rewrites.

## Swift change in HermitPrompt.swift
1. `prepare_swap`: no change.
2. Append these four lines at the end of `extraToolLines` (after the `prepare_withdraw` line, inside the `#"""` ... `"""#` string, same 8-space indent,
   no trailing comma or blank line). Generated with `json.dumps(t, ensure_ascii=False)`:

```swift
        {"type": "function", "function": {"name": "prepare_buy", "description": "Prepare a buy of a Solana token with USDC or SOL through Jupiter and open the Buy review for the user to approve. Only Jupiter-verified tokens. Nothing is executed by this call.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "Token ticker, e.g. 'WIF', 'BONK'."}, "amount_usd": {"type": "number", "description": "How much to spend, in US dollars."}, "pay_with": {"type": "string", "enum": ["USDC", "SOL"], "description": "Optional. What to pay with. Leave out unless the user said; the app uses USDC if held, else SOL."}}, "required": ["symbol", "amount_usd"]}}}
        {"type": "function", "function": {"name": "prepare_close", "description": "Prepare closing an open Hyperliquid position, fully or by percent (market, reduce only), and open the Close review for the user to approve. Nothing is executed by this call.", "parameters": {"type": "object", "properties": {"coin": {"type": "string", "description": "Ticker of the open position, e.g. 'SOL'."}, "percent": {"type": "number", "description": "Optional. Share of the position to close, 1 to 100. Leave out to close all of it."}}, "required": ["coin"]}}}
        {"type": "function", "function": {"name": "get_activity", "description": "Get the user's recent wallet activity: sends, swaps and buys with status (confirmed, failed or in progress), newest first, and how many are in progress. Use it to check whether something happened.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Optional. How many rows to return, newest first. Defaults to 10."}}, "required": []}}}
        {"type": "function", "function": {"name": "open_app", "description": "Open a site or dApp in Hermit's browser, by name (e.g. 'Uniswap', 'Aave') or by url. Nothing connects to the wallet until the user approves it.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Name of a dApp or site, e.g. 'Uniswap'."}, "url": {"type": "string", "description": "A full web address, e.g. 'https://app.uniswap.org'. Use this when the user gave one."}}, "required": []}}}
```

3. Update the doc comment above `extraToolLines` ("NEAR Intents quotes, batch sends, Aave deposits and withdrawals, Jupiter buys, closes, activity, open app"),
   `testAppToolsAreTheTrainedOnesPlusQuote` (19 names) and regenerate `render_golden.json` with 19 lines (currently 11).

## Not byte differences, but app/training gaps to fix (app side)
- `ToolLoop.prompt(tools: false)` (ToolLoop.swift:61) passes `[]`; `HermitPrompt.render` treats `[]` as non-nil and emits an empty
  `<tools>\n</tools>` block that training never has. Fix: pass `nil` (`tools ? config.tools : nil`) or make `render` use `tools?.isEmpty == false`.
- `ToolLoop.prompt` replaces "Face ID" in the whole text (also user text / tool results) for Touch ID devices; assemble4 does the same on every message
  (line ~291), so they agree. Tool-call arguments are not replaced in either.
- Number text: the app keeps numbers as the model wrote them, training rows contain `100.0`-style floats (114 calls: stop_loss, entry, take_profit, limit_price)
  and one `9.3e-05`; this is only a model-output habit, not a render mismatch.
- Mid-conversation system messages render in the app but are skipped by the lab; none exist in data3.
