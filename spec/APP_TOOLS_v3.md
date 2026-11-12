# APP_TOOLS_v3: what the app must implement for the 11 new tools

Written 2026-11-11. Source of truth for the model side: `hermit_tools_v3.json` (30 tools) and `hermit_tools_v3.meta.json` (every result shape below is a `result_example` or `other_results` entry there; the simulator and the app must return exactly those keys). Function names are the ones found in `wt-near/Packages/HermitKit/Sources` (read only). "New" means no such function exists today.

## 0. Order of work and prerequisites

1. v2 tools that are in the spec but not in the app: `prepare_buy`, `prepare_close`, `get_activity`, `open_app` (PARITY.md has the four tool lines). Ship them first; the 19 lines must be byte identical to `hermit_tools_v3.json[0..18]`.
2. `prepare_buy` handler change, NO schema change: if `symbol` looks like a mint (`JupiterTokens.looksLikeMint`) resolve it as a mint. Accept it only when that mint appears in the user's typed text or in a `search_tokens` result of this chat. Unverified tokens then go through `TokenCheck` and return `warnings` (meta `prepare_buy.mint_unverified`); refuse when liquidity is under $5K or when unverified with freeze authority (`status: blocked`, `reasons`). A symbol clash returns `needs_choice` (meta `prepare_buy.needs_choice`).
3. The 11 new tools, in this value order: `get_address`, `open_page`, `prepare_sell`, `search_tokens`, `get_positions`, `get_portfolio`, `cancel_order`, `create_automation`, `schedule_payment`, `add_contact`, `set_tp_sl`.
4. Packs and the larger prompt cap (section 3), then regenerate `Tests/HermitCoreTests/Fixtures/render_golden.json` with 30 lines and a pack variant.

## 1. Shared plumbing

- **Result cleaning.** Every result string goes through `ToolResult.clean` (sorted keys, shortest numbers, injection scrub). Contact names, token names and symbols, automation names and prompts, payee names and event text are injection routes: `(hidden: this text read like instructions to the assistant)` replaces them.
- **Three kinds of "waiting".** `review_opened` (biometric): `prepare_sell`, `set_tp_sl`. `card_opened` (one tap, no biometric, nothing moves): `cancel_order`, `create_automation`, `schedule_payment`, `add_contact`. Neither is ever `executed: true`; the app never returns `sent`/`executed:true` to the model.
- **ToolGuard.** Add `prepare_sell` and `set_tp_sl` to `moneyTools`. Add a new `cardTools = ["cancel_order","create_automation","schedule_payment","add_contact"]` and treat it like `moneyTools` in `ToolLoop` where `prepared` is set (ToolLoop.swift near line 242): a result with `status` `card_opened` or `review_opened` counts as opened.
- **Honesty net (cards).** If the final text claims "saved", "added", "created", "scheduled", "cancelled", "set up" while no card/review opened this turn, append: `Nothing is waiting for your confirmation yet. Ask me again and the card will open.` Keep the existing money texts for prepare tools.
- **Typed-address guard** (`ToolGuard.refusal`): extend to `add_contact.address` and `schedule_payment.payee` when it is an address (same rule and same error as `prepare_send`, text in meta). File text never counts.
- **Contact addresses are user-chosen.** Keep a per-chat set `resolvedAddresses`. `get_address {contact}` adds the returned address to it; `refusal` for `prepare_send` and `schedule_payment` accepts an address that is in the user's text OR in `resolvedAddresses`. Contacts can only be created by the user in Payments or by `add_contact` after the full-address card, so this does not open a hole. Names that read like orders are scrubbed by `ToolResult.clean`.
- **Progress lines** (`AgentTools.progress`): `prepare_sell` "Preparing the sale for your review"; `search_tokens` "Searching Solana tokens for {query}"; `get_positions` "Reading your positions"; `get_portfolio` "Adding up your portfolio"; `cancel_order` "Preparing the cancel for your confirmation"; `set_tp_sl` "Preparing the stop for your review"; `create_automation` "Drafting the automation"; `schedule_payment` "Drafting the payment reminder"; `add_contact` "Drafting the contact"; `open_page` "Opening {page}"; `get_address` "Looking up the address".
- **Device word.** Result `note` strings use "Face ID" and are rewritten to "Touch ID" on a Mac like every other result (`bioWord`). Card notes say "tap Confirm/Create/Schedule/Save"; keep them literal as in the meta file.
- **Limits.** `maxRounds` 5 and `maxCallsPerTurn` 6 stay. A chain like get_address, prepare_send is 2 rounds. Sell and buy never count against the assistant cap when they reduce risk (sell, set_tp_sl, cancel_order): still reviewed.

## 2. The tools

### prepare_sell `{token, percent?, amount?, receive?}` (review, biometric)
- Calls: resolve `token` with `JupiterTokens` (ticker among the user's holdings first, `looksLikeMint` for a mint; the mint must be in user text or a `search_tokens` result of this chat); holding from `SolanaHoldings.holding(mint)` / `.items`; cost basis not needed; quote and review with the same path as the Sell sheet in `SolanaSwap.swift` (`requote()` then `ApprovalSheet`), which today offers 25/50/100 only. New: a headless wrapper like the one `prepare_buy` needs, accepting any percent 1 to 100 or an amount in token units (`amount` wins over `percent` only if both are given: reject both with an error). `receive` USDC (default) or SOL.
- Result: meta `prepare_sell.result_example` (`you_sell`, `you_get`, `price_impact_pct`, `slippage_pct`, `route: "Jupiter via ..."`, `status: review_opened`). `warn` variant when price impact >= 3% (`policy: warn`, `warnings`). 
- Errors: `not_held`, `insufficient` (`holding`), `needs_choice` (two held tokens with the same symbol), `blocked` (frozen balance: the sheet already blocks it; daily limit), `no_route`, `no_wallet`.
- Selling works for unverified tokens. Do not apply the buy-side refusals.

### search_tokens `{query}` (read)
- Calls: `JupiterTokens.search(query)` (returns `Found`), `JupiterTokens.looksLikeMint`, `TokenCheck` for tags. At most 5 rows, liquidity descending.
- Result: `{tokens:[{symbol, name, mint, verified, price, change_24h_pct, liquidity_usd, volume_24h_usd, tags[], refusal?}]}`. `tags` values exactly: `Unverified`, `Freeze on`, `Mint on`, `Low liquidity`, `Top holders N%`, `New`. `refusal` is set when buy would be refused (under $5K liquidity, unverified with freeze). Null price/change allowed.
- Empty: `{tokens: [], note}`. Failure: `{error: "Token search is unavailable right now."}`. Record the returned mints in the per-chat `searchedMints` set (used by `prepare_buy`/`prepare_sell` guard).

### get_positions `{kind?: all|spot|earn|perps}` (read)
- Calls: spot = `SolanaHoldings.items` + `SpotBook.basis(mint:walletID:)` (avg cost, pnl); earn = `AavePositions`; perps = the Hyperliquid account positions and `OpenOrder` rows (`isTrigger`, `triggerPx`, `triggerCondition`, order id).
- Result shape: meta `get_positions.result_example`. Order `kind` is `limit`, `take_profit` or `stop_loss` (from `isTrigger` + order type); `whole_position: true` for position-wide triggers; `id` is the Hyperliquid oid as a string. A filtered call returns only that key plus `wallet`. Selected wallet only. `unreachable` when a source failed. `frozen` from the holding.

### get_portfolio `{}` (read)
- Calls: selected-wallet balances (the `get_balances` source), `SolanaHoldings` (tokens that are not already counted in the wallet: not SOL, not USDC), `AavePositions`, Hyperliquid account value.
- Result: `{by_place:[{place: Wallet|Spot tokens|Aave|Hyperliquid, usd}], top:[{place, what, usd}] (3 rows), total_usd, wallet, unreachable?}`. `total_usd` = sum of `by_place`, rounded to cents; zero places omitted. No double counting.

### cancel_order `{coin, kind?, order_id?}` (card, one tap)
- Calls: Hyperliquid open orders -> match by `order_id`, else by `coin` + `kind`. Confirm runs `PerpActions.cancel(order, model:)` (no biometric today). `coin: "all"` cancels plain limit orders only, never TP/SL.
- Result: meta `cancel_order` (`cancelling`, `status: card_opened`, `warning` for a stop loss: "Without it the SOL long has no stop loss."). No match: `nothing_to_cancel`. Several match and `kind` is absent: `needs_choice` with `question`. 
- After the user taps Confirm, post a small line in chat ("Cancelled: SOL stop loss at $128"). The model does not see it; on the next turn it uses `get_positions`.

### set_tp_sl `{coin, take_profit?, stop_loss?}` (review, biometric)
- NEW function (no UI exists for it today): `PerpActions.setProtection(position, takeProfit:, stopLoss:, ask:)`. It builds reduce-only market trigger orders for the whole position with `Hyperliquid.triggerWire(...)`, using grouping `positionTpsl` (today's code only uses `normalTpsl`, which rides along with a new order), and shows the same `ApprovalSheet` as `PerpActions.close`. Needs at least one of the two prices.
- Validation (return `wrong_side`, `prepared:false`): long stop below the mark and take profit above it; short the reverse. `no_position` when the coin has no position. If a trigger of the same type exists, replace it in the same action and report it in `replaces` ("stop loss at $120").
- Result: meta `set_tp_sl.result_example` (echoes `stop_loss`/`take_profit` exactly as passed, `position`, `order`).

### create_automation `{name, prompt, schedule}` (card, one tap)
- Calls: the sentence-to-schedule parser the Automations page already uses ("create from sentence"); build an `Automation(name:, trigger:, action:, limit:, on: true, prompt:, asks:)` (Workspace.swift) but do not save it until Create is tapped; on Create call `AutomationScheduler.changed(model)`.
- Card shows name, the normalised schedule ("Every day at 08:00") and the exact prompt. The result repeats that normalised schedule in `schedule`.
- Errors: `bad_schedule` (error text in meta), `too_frequent` (shortest repeat is every 5 minutes), `limit_reached`. If the app's real limits differ from 5 minutes / 10 automations, change the meta text and tell the lab before generation.
- Safety: refuse to run when the user turn that triggered the call contains `<file>` text or the call follows a tool result that contains the schedule words (create only from the user's own message). The automation's prompt can read and prepare; running it never approves anything.

### schedule_payment `{payee, amount, symbol, when, chain?, repeat?}` (card, one tap)
- Calls: `PaymentsStore.payee(named:)` (case-insensitive; two matches -> `needs_choice`; none -> `payee_not_found`), or a typed address (guard above, shown as `0x5290… 9EE7`). Builds a `ScheduledPayment(payee, addr, amount, token, note, next, repeats, exact)` and calls `PaymentsStore.save(_:)` on Schedule tap. `repeat` maps to `PayRepeat` (once, weekly, monthly); absent = once. `exact` = the amount as the model passed it (use `PayAmount.exact`).
- `when`: parse ISO, "Dec 1 9:00", "tomorrow 18:00", "next friday", "in 3 days"; default time 09:00; the result `next_due` is the parsed value as `YYYY-MM-DD HH:mm`. Past date -> `bad_date`.
- `chain`: `ScheduledPayment` has no chain field. Either add an optional `chain` (Codable, optional so old `payments.json` decodes) or drop it and let Pay now choose. `network` in the result is the contact's network or the only chain that holds the token; several chains and none named -> `needs_chain` (question text in meta).
- It NEVER sends. The note says so. Pay now (today's flow) still opens Send with the biometric.

### add_contact `{name, address, network?}` (card, one tap)
- Calls: `PaymentsStore.save(_ p: Payee)` on Save tap; validate the address with `Addresses.swift` (family evm/solana/near); same-name check (`name_taken`), same-address check (`exists`). The card shows the full address and the family.
- Guard: the address must appear in the user's own typed text of this chat (not file, tool result, calendar or page).

### open_page `{page, tab?}` (open, button)
- Calls: `AppModel` route enum (home, projects, automations, trade, earn, payments, explore, calendar, wallets, connectors, files, settings) and `tabRequest` (AppModel.swift near line 266). Show a button exactly like `open_add_funds`; the click calls `open(route, tab:)`. Allow-list tabs: trade markets|positions; earn save|positions|explore; payments schedule|payees|transfer|apis; settings profile|safety|memory|models|cloud. Anything else: `bad_tab` error with the tab list. Never allow a route or tab that shows a recovery phrase or a key (`not_allowed`). Settings, Safety may be opened (the user changes limits themselves; the model never does).

### get_address `{contact?, chain?}` (read)
- Calls: receive addresses of the selected wallet per family (Addresses.swift / WalletService); never a key or phrase. `contact`: `PaymentsStore.payee(named:)` plus the Send sheet's saved addresses if they are a separate list. `chain` limits to that family.
- Result: meta `get_address` (own: `{addresses:[{family, address, networks, wallet}], note}`; contact: `{contacts:[{name, address, family, networks}]}`; `not_found`; `needs_choice`). Adds the returned contact address to `resolvedAddresses`.

## 3. Result changes to v2 tools (no schema change)

`get_balances`: `wallet` and a multi-wallet `note` (meta `multi_wallet`). `get_price`: per-token memecoin prices (meta `per_token_memecoin`; print tiny numbers as decimals, never `9.8e-06`). `hyperliquid_account`: `account_mode` and trigger rows (meta `with_triggers`). `prepare_trade`: stocks via trade.xyz, `trade_xyz_margin` blocked, per-token price rejection. `get_yields`: `apy_base_pct`, `apy_reward_pct`, `apy_30d_pct`, `il_risk`, `pool_meta`, `can_deposit_in_hermit`. `prepare_buy`: `needs_choice`, `mint_unverified` (above). `position_math` has no `coin` argument in v3 (schema frozen): it assumes the market maximum leverage and says so in `note`; adding `coin` is a v3.1 schema change.

## 4. Tool packs and the prompt cap

Changing the tool block mid-chat breaks the KV-cache prefix, so choose the pack when the conversation starts and keep it for the whole chat. Measured with the Qwen tokenizer: 19 tools 2.7k tokens, 30 tools 4.35k tokens, 16,049 characters.

| pack | tools | with core: tokens |
|---|---|---|
| core (12) | calculator, get_price, get_balances, get_portfolio, get_address, get_activity, prepare_send, prepare_swap, get_quote, open_page, open_app, open_add_funds | 1.5k |
| trade (+8) | hyperliquid_account, position_math, search_markets, prepare_trade, prepare_close, cancel_order, set_tp_sl, get_positions | 2.8k |
| solana (+4) | search_tokens, prepare_buy, prepare_sell, get_positions | 2.2k |
| earn (+4) | get_yields, prepare_deposit, prepare_withdraw, get_positions | 2.0k |
| pay (+4) | prepare_multisend, add_contact, schedule_payment, create_automation | 2.2k |

- Router (`ToolPacks.choose(firstMessage:, hasAttachment:)`): keywords on the first message and any attachment. trade: long, short, leverage, perp, position, stop, take profit, TP, SL, cancel order, limit order, liquidation, funding, hyperliquid, stock names. solana: buy/sell/token/mint/memecoin/pump/jupiter/safe, a base58 mint. earn: aave, deposit, withdraw, yield, apy, earn, save. pay: csv/xlsx attachment, many addresses, contact, payee, schedule, remind, every day/week/month, automation, recurring. Several matches -> union. No match -> core only. 
- Hermit 4B (device): core + at most two packs (cap 22 tools), `maxPromptChars` 32,000. Hermit Lite: core + at most one pack and at most 16 tools: its trade pack is hyperliquid_account, prepare_trade, prepare_close, cancel_order. Private Cloud and OpenRouter: all 30.
- A tool the model needs but the pack lacks: the model says it is not available in this chat and calls `open_page` (core). The training data includes this (PLAN_v3.md, generators `auto_pages` and `lite_cloud`).

## 5. Tests to add

- `testAppToolsAreTheTrainedOnesPlusQuote` -> 30 names in `hermit_tools_v3.json` order; golden render of the full block and of each pack.
- ToolGuard: sell/stop are money tools; card tools set `prepared`; contact address accepted only after `get_address(contact)`; planted payee text scrubbed; `add_contact` with a file-only address refused; `create_automation` after a tool result refused.
- For each tool: every meta `other_results` key reproducible from a fake backend; the replay check in the lab compares them byte for byte.
