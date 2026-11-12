---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: mlx
tags: [mlx, hermit, tool-calling, on-device, near]
---
# Hermit 1.3

Hermit is a wallet for iPhone and Mac with an AI assistant that runs on the device. You ask it in plain words to swap, bridge, trade or pay. It prepares the move, and nothing leaves your wallet until you approve it with Face ID or Touch ID. We didn't want our balances and plans on somebody's server, so the model lives on your device. This is that model.

## Try it

- Mac: https://hermit.my/dmg
- iPhone beta: https://testflight.apple.com/join/ed9vUh7d
- Mac beta: https://testflight.apple.com/join/P66H5DF8
- Docs: https://docs.hermit.my
- Code: https://github.com/Hermit-Ai-Labs/hermit-app

The app downloads this model on first launch and checks every file against hashes built into the app. You don't need to download anything from here yourself.

## Things you can ask it

- "Swap 20 USDC on Base to NEAR" opens the Swap review with a NEAR Intents quote.
- "Sell half my BONK" opens a Sell review for your Solana tokens.
- "Long ETH with $50 at 5x" gives you a review for a Hyperliquid order.
- "What are my positions?" reads them. "Cancel my DOGE stop" asks first.
- "Every morning at 8, brief me on my portfolio" creates an automation.
- Turn off Wi-Fi and it still answers.

![Hermit at a glance](https://raw.githubusercontent.com/Hermit-Ai-Labs/hermit-app/main/docs/img/system_overview.png)

## How we trained it

We trained it ourselves. It's a LoRA fine-tune of Qwen3-4B-Instruct-2507 (Apache-2.0) on 21,000 examples of real Hermit tasks: all 30 tools, multi-step requests, recovering when a quote fails, and saying no to anything unsafe. Two passes on a rented NVIDIA L40S. The tools are sent in small packs, so the prompt stays short and replies start fast.

This repo is the MLX 5-bit build (about 2.6 GB). We checked the formats: 5-bit agreed with the full-precision model on 37 of 40 first steps, 4-bit on 34.

## How it did

After the first pass, on 1,801 test conversations:

- Never claimed something was done without a tool result to back it: 1,801 of 1,801
- Never asked for a recovery phrase or key: 60 of 60
- Exact amounts and addresses on the new tools: 78 to 87%
- Multi-step requests finished: 62%. Follow-ups right: 75%

The weights here have the second pass on top. Full two-pass scores and a side-by-side with Hermit 1.2 come next, with a training round aimed at multi-step requests, follow-ups and knowing when not to call a tool.

Smaller iPhones use [Hermit 1.3 Lite](https://huggingface.co/aiquantgod/Hermit-1.3-Lite-MLX-6bit).

## What it can and can't do

It can only call Hermit's tools. It never holds a key. Anything that moves money goes through the app's spending policy and a review you approve, and only then do the keys sign. More in the [security notes](https://github.com/Hermit-Ai-Labs/hermit-app/blob/main/docs/SECURITY-DEMO.md).
