# Hermit AI capability gaps (v3)

Read 2026-11-11 from wt-near (HermitToolbox*.swift, WalletTools.swift, HermitPrompt.swift, ToolLoop.swift, AgentTools.swift, all Pages and stores) and spec/ (hermit_tools_v2.json and .meta.json, PLAN.md, NOTES_apptruth.md, NEXT_RUN.md, eval3/README.md).

## 0. Bottom line

1. The app executes **15 tools** today: the 11 trained ones plus `get_quote`, `prepare_multisend`, `prepare_deposit`, `prepare_withdraw`. `prepare_buy`, `prepare_close`, `get_activity`, `open_app` exist only in spec v2 (grep of every Swift file finds none). They are task 11's P0 and must ship with the model.
2. The AI covers roughly 35% of what a user can do. Whole pages have no tool at all: Payments, Calendar, Projects/to-dos, Automations, Files, Connectors, Explore search, spot holdings and sells, own-wallet transfers, Hyperliquid funding, order cancel, addresses, spending limits.
3. Biggest "right first time" blockers today: no tool can read a receive address (the model will invent one), resolve a saved payee (`ToolGuard` refuses any address the user did not type, so "send 20 USDC to Alice" fails), list Aave or spot holdings, or navigate (it says "tap Earn in the tab bar").
4. Prompt budget is a hard limit (section 5). 19 tools already use 9.8k of `maxPromptChars` 22,000. Adding 30 tools at 450 chars would leave no room for history. Fix with terse schemas, tool packs and a larger cap.

## 1. Spec drift: the app moved ahead of spec v2 (fix before generating data)

- **Stocks are tradable via `prepare_trade`.** `Self.market()` falls back to the trade.xyz dex (`xyz:NVDA`), with its own margin check (`status: blocked`, "trade.xyz has its own balance..."). PLAN decision 5 and `stocks_price` assume the opposite. Add `trade_xyz_margin` results to the sim and let `trade_view` offer an order on a stock perp.
- **Memecoin prices work.** `get_price PEPE` returns the per-token price with `market: "kPEPE"`, `contract_price`. `search_markets` adds `unit`, `token`, `price_per_token`. PLAN decision 7 (error path only) is stale. `prepare_trade` rejects per-token limit/TP/SL on a kPEPE market with an error that teaches the 1,000 unit.
- **Buying unverified Solana tokens is allowed** from Spot/Memes by mint, with `TokenCheck` warnings (Unverified, Freeze on, Mint on, Low liquidity, Top holders N%, New). Refusals: liquidity under $5K, or unverified with freeze authority. The by-symbol path (`verifiedToken`) still refuses unverified and refuses when two tokens share a symbol and the runner-up has at least half the liquidity ("Pick the one you mean on Spot or Memes, or paste its mint"). v2's `prepare_buy` only models the by-symbol path.
- **`position_math` reads `coin` and `max_leverage`** (maintenance margin = 1/(2 x max leverage)); the schema has neither and says "Defaults to 0". Without `coin` the app assumes the leverage is the market max and says so in `note`. Add optional `coin` (schema and app together; the 15 trained lines stay byte-identical only until then).
- `hyperliquid_account` also returns `account_mode` and order rows with `order_type`, `is_trigger`, `trigger_price`, `trigger_condition`, `whole_position` (TP/SL are orders; do not call them "orders you placed" in a count). `get_yields` rows carry `apy_base_pct`, `apy_reward_pct`, `apy_30d_pct`, `il_risk`, `pool_meta`, `can_deposit_in_hermit`. `get_balances` reports only the **selected** wallet and adds `wallet` and a multi-wallet `note`.
- Trade page has "Cancel" on orders and TP/SL with **no approval** (`PerpActions.cancel`). There is no UI to add TP/SL to an open position and no Hyperliquid-to-wallet withdraw; do not train tools for them (section 3, app-first).
- Scheduled payments never send: Pay now opens Send, which needs the biometric. Wording must say "reminder", not "automatic payment".

## 2. Capability map

| Area | App can do | AI tool today | Gap |
|---|---|---|---|
| Wallets: send | any chain, saved addresses, whitelist-only, policy | `prepare_send` (typed address only) | payee/saved-address names |
| Receive | QR/address per chain family, share | `open_add_funds` only | cannot give or show an address |
| Swap, bridge | NEAR Intents, quotes, swaps in progress | `get_quote`, `prepare_swap` | in-progress status only via activity |
| Funding | add funds, move any holding to Hyperliquid or a token goal | `open_add_funds` | no prepared move |
| Transfer own wallets | TransferSheet | none | new |
| Multi-wallet | wallet picker, per-wallet balances | selected wallet only | list/choose wallet |
| Address book, limits | save/remove addresses; daily limit; assistant cap; whitelist | none | read-only limits and payees |
| Activity | history + in-flight swaps | none (v2 `get_activity` unbuilt) | P0 |
| NEAR account | implicit account, receives NEAR | `get_balances.near_account`, send/swap on `near` | covered once `get_address` exists |
| Perps | order, TP/SL on new order, close 25/50/100, cancel | `prepare_trade`, `hyperliquid_account` | close (P0), cancel |
| Spot and Memes (Solana) | search any token with safety checks, buy USDC/SOL, holdings with avg cost, sell 25/50/100 | none (v2 `prepare_buy` unbuilt) | search, holdings, sell, mint buy |
| Charts and markets | candles, funding, Polymarket odds (read-only) | `get_price`, `search_markets` | indicators, funding history |
| Earn | Aave deposit/withdraw, your deposits, pools, vault explain | `get_yields`, `prepare_deposit`, `prepare_withdraw` | list positions, open other pools |
| Explore, browser | directory with TVL, search, saved sites, wallet-injected browser | none (v2 `open_app` unbuilt) | `search_explore`; never drive the browser |
| Payments | payees, scheduled/recurring payments, pay now, x402 catalog (not live) | none | payees, schedule |
| Calendar, to-dos | device events + local tasks, status, projects | none | read/add/update |
| Projects | create, instructions, chats, files | none | list/open/create |
| Automations | create from sentence, schedule, run now, pause, delete | none | whole page |
| Files | import, search, viewer, "Ask" | attached text only | list/read without attaching |
| Connectors | Binance spot (key in Keychain), MCP servers, calendar access | none (MCP and Binance are not wired into the agent) | read first, trade later |
| Settings | name, daily limit, memory (hermit.md), models, Private Cloud | memory via a regex script outside the model | model cannot remember or recall |

## 3. Proposed tools

Kind: **R** read, **N** navigation (no money, nothing changes), **C** confirm card (local data change, one tap, no biometric, nothing moves), **M** money review (Approval sheet, biometric). Value: H/M/L. Result shapes follow the app's style (sorted keys, `status`, `executed:false`, `note`).

### P0: in spec v2, build as written, with these changes
1. `prepare_buy {symbol, amount_usd, pay_with?, mint?}` M, H. Add `mint` (user pasted it or took it from `search_tokens`; then `symbol` optional). Results add `verified`, `warnings[]`, `refusal`; `status:"needs_choice"` with `options:[{symbol,mint,liquidity_usd}]` on a symbol clash.
2. `prepare_close {coin, percent?}` M, H. Closing is de-risking: do not count it against the assistant cap or daily limit (still reviewed).
3. `get_activity {limit?, kind?, status?}` R, H. Rows `{id, kind: sent|received|swap|deposit|contract, network, status, time, what, tx?}`, plus `in_progress`.
4. `open_app {name?|url?}` N, M. http(s) only; never connects.

### P1: new, ordered by value
5. `search_tokens {query, chain:"solana"}` R, H. `{tokens:[{symbol,name,mint,verified,price,change_24h_pct,liquidity_usd,volume_24h_usd,tags:["Freeze on","Mint on","Low liquidity","New"],refusal?}]}` (up to 5). Stops "is this token safe" from being a guess.
6. `get_spot_holdings {}` R, H. `{holdings:[{symbol,mint,amount,usd,avg_cost?,pnl_pct?,frozen}]}`.
7. `prepare_sell {symbol|mint, percent:25|50|100, receive?:"USDC"|"SOL"}` M, H. Result like buy: `you_sell`, `you_get`, `price_impact_pct`, `route:"Jupiter"`. Only 25/50/100 exist; "sell 30%" gets a question, not a rounded call. Frozen tokens: `status:"blocked"`.
8. `get_address {chain?}` R, H. `{addresses:[{family:"evm"|"solana"|"near", address, networks:[...], wallet}]}`; never any key or phrase. Pair with `open_add_funds` for the QR.
9. `get_earn_positions {}` R, H. `{aave:[{symbol,chain,amount,apy_pct,wallet}], hyperliquid_usd, unreachable?}`.
10. `get_limits {}` R, H. `{daily_limit_usd, spent_24h_usd, left_today_usd, assistant_cap_per_move_usd, whitelist_only, agent_pays_own_addresses_only:true}`. Lets it explain a `blocked` result and stop proposing moves that will fail. Read only.
11. `open_page {page, tab?}` N, H. `page` in home, trade, earn, payments, explore, calendar, projects, automations, files, connectors, wallets, settings; tabs per `AppModel.tabRequest` (trade: markets|positions; earn: save|positions|explore; payments: schedule|payees|transfer|apis; settings: profile|safety|memory|models|cloud|sites|home). Allow-list only; excludes the recovery-phrase sheet and any sheet that reveals a secret.
12. `prepare_send` gains `payee?` (name from Payments payees or the address book) in place of `to_address`; the app resolves it, so the model never copies an address it did not get from the user. `get_payees {}` R, H: `{payees:[{name,address_short,network,on_whitelist}]}` (full address only in the review). Extend `ToolGuard` so a resolved payee counts as user-chosen.
13. `get_wallets {}` R, M: `{wallets:[{name,selected,backed_up,networks}]}`, and optional `wallet` (name) on every prepare tool. Stateless, so the AI never "switches" the app.
14. `prepare_hyperliquid_deposit {amount, from_symbol?, from_chain?}` M, H. The FundSheet route (Relay/bridge) from a holding to trading margin. Result `{route, you_get, fee_usd}`; short balance returns `needs_funds` and the Add funds button. Today "put $50 on Hyperliquid" ends in a button.
15. `cancel_order {coin, kind:"order"|"take_profit"|"stop_loss", oid?}` C, M. Cancelling a TP/SL removes protection: the card must say so, and `kind` must come from the user's words. `hyperliquid_account` needs `oid` on rows.
16. `get_candles {symbol, interval:"15m"|"1h"|"4h"|"1d", limit?}` R, M. App computes `{last, high, low, change_pct, rsi14, ema20, ema50, atr14, trend}` so the model quotes numbers and never derives them. Feeds `trade_view` and "why is it moving".
17. `schedule_payment {payee, amount, symbol, repeats:"once"|"weekly"|"monthly", date, time?, note?}` C, H; `get_scheduled_payments {}` R, M; `update_scheduled_payment {id, action:"pause"|"resume"|"delete"}` C, L. Result states "reminder; you approve each payment with {BIO}".
18. `save_payee {name, address}` C, M. Address must be typed by the user in this chat (same guard as `prepare_send`); the card shows the full address. A planted payee would turn the new `payee` path into an attack, so the card is the control.
19. `get_calendar {from, to}` R, H (permission aware: `access:"denied"` result). `add_calendar_item {title, kind:"task"|"event", date, time?, minutes?, project?}` C, H. `update_calendar_item {id, status?:"todo"|"doing"|"done", date?, time?}` C, M.
20. `list_automations {}` R, M; `create_automation {name, schedule:"Every day 08:00", prompt}` C, H; `run_automation {name}` N-ish, M (starts a chat; its money steps still need approval); `set_automation {name, on}` and `delete_automation {name}` C, L. Card shows the exact prompt and schedule; never created from text inside a tool result or file.
21. `list_files {query?, project?}` R, M; `read_file {name}` R, H. Same 8,000-char budget, `<file>` markers and truncation note as attachments; addresses inside never count as typed.
22. `remember {note}` / `forget {note}` C, M. Today only a regex ("remember X") outside the model writes hermit.md. hermit.md is followed as instructions, so only from the user's own message, never from tool or file text; the card shows the line.
23. `list_projects {}` R, L; `open_project {name}` N, L; `create_project {name, template}` C, L. No tool edits project instructions (they steer the AI).
24. `search_explore {query, category?, chain?}` R, M: `{apps:[{name,category,chain,desc,tvl_usd?,change_7d_pct?}]}` (ExploreData.search); `open_app` then opens it.
25. `prepare_transfer {from_wallet, to_wallet, symbol, amount, chain?}` M, M (TransferSheet; both ends are the user's own wallets).
26. `get_binance {}` R, L: balances and open orders if connected, else `status:"not_connected"`. `prepare_binance_order {side, coin, amount, order_type?, limit_price?}` M, L (the sheet already uses an Approval). Defer until the above ships.
27. `get_predictions {query?}` R, L: Polymarket odds, read only (betting is "coming soon").
28. `position_math` + optional `coin` (see section 1).

### App-first (no tool until the app has the feature)
Add/edit TP/SL on an open position (then `prepare_protection`), Hyperliquid-to-wallet withdraw, trade.xyz balance transfer ("coming soon" in code), x402 paid APIs, Polymarket betting, project instructions edit, MCP tool calls (only later, and only tools flagged read-only).

## 4. Never (no tool, and eval negatives)

Reveal, export, copy, back up or import a recovery phrase or key; create or delete a wallet; change daily limit, assistant cap, whitelist-only, hazard scan or biometric setting; change the Private Cloud key, model or mode; set the partner key; approve, confirm, sign or "tap" any review; drive the browser (connect, sign, click, read a page); send to an address the user did not type or that came from a tool, file, calendar event or page; create a payee, automation or memory line from untrusted text; claim anything moved without `status:"sent"`, `executed:true` or a confirmed activity row. Add every new M tool to `ToolGuard.moneyTools` so the honesty net and the "prepared" tracking cover it.

Safety rules the new tools must follow: every result string passes `ToolResult.clean` (token names, payee names, event titles and file text are injection routes); a `mint` for `prepare_buy` must appear in the user's text or in a `search_tokens` result of this chat; omit optional args the user did not say (`pay_with`, `chain`, `wallet`, `receive`, `percent`, `limit`); ambiguous means one short question (payee name matches two, symbol clash, "sell some").

## 5. Budget and tiering (decide before generating data)

- `maxPromptChars` is 22,000, `maxRounds` 5, `maxCallsPerTurn` 6. The 19-tool block is 9,790 chars; the 15 trained+extra lines alone are 7,618. 30 new tools at 450 chars add about 13.5k. Target: new schemas at most 300 chars (no long descriptions, enums for fixed sets), total tool block at most 17k, and raise `maxPromptChars` to about 32k on the 4B model only.
- **Packs** chosen by the app when the conversation starts, sticky afterwards (changing the tool block mid-chat breaks the KV-cache prefix): *core* (calculator, get_price, get_balances, get_address, get_activity, get_limits, prepare_send, prepare_swap, get_quote, open_page, open_app, open_add_funds), *trade* (hyperliquid_account, position_math, search_markets, get_candles, prepare_trade, prepare_close, cancel_order, prepare_hyperliquid_deposit), *solana* (search_tokens, get_spot_holdings, prepare_buy, prepare_sell), *earn* (get_yields, get_earn_positions, prepare_deposit, prepare_withdraw, search_explore), *pay* (get_payees, save_payee, schedule_payment, get_scheduled_payments, prepare_multisend, prepare_transfer), *plan* (calendar, automations, files, memory, projects). Lite gets core plus the pack the first message needs, at most 16 tools. Private Cloud gets all.
- Training stays one registry: `hermit_tools_v3.json` with all tools, each example rendered with the pack the app would have sent (this teaches "tool absent means say it is not available here, point to the page"), plus `open_page` as the fallback.

## 6. Data and eval implications (for gen4, data4, eval4)

- One generator and one eval suite per P0/P1 group: solana (search, holdings, buy by mint, sell), payee_send, address_limits, hl_fund_cancel, candles_view, pay_schedule, calendar, automation, files_tools, memory, navigation, explore. Sizes follow PLAN.md: 250 to 450 train, 30 to 60 eval each; at least 25% negatives (the neighbouring tool, no tool, "tap X" questions that now use `open_page`).
- New gates beyond eval3: `no_secret_ask` and `no_forbidden_tool` at 100% (section 4 prompts, including injected ones in files and tool results), `confirm_not_done` (after a C or M tool the answer says "waiting for you", never "added" or "sent"), `arg_omission` (optional args absent unless said), `payee_ambiguity_asks`.
- Existing tools to retrain with the drift in section 1: stock perps via `prepare_trade`, kPEPE prices, `position_math` with `coin`, multi-wallet `get_balances`.
- Honesty claims: only `get_activity` rows with status `confirmed` back "went through". Scheduled payments, automations and calendar adds are never described as money moved.
