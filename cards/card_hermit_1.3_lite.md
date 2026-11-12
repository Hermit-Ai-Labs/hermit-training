---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: mlx
tags: [mlx, hermit, tool-calling, on-device, near]
---
# Hermit 1.3 Lite

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

We trained it ourselves. It's a LoRA fine-tune of Qwen3-1.7B (Apache-2.0) on 21,000 examples of real Hermit tasks, two passes on one rented NVIDIA A6000. It's the model for iPhones that don't have the memory for [Hermit 1.3](https://huggingface.co/aiquantgod/Hermit-1.3-4B-MLX-5bit), and it knows the same 30 tools, in smaller packs.

This repo is the MLX 6-bit build (about 1.4 GB). 4-bit lost too much: 26 of 40 first steps right, against 36 at full precision. 6-bit gets 35.

## How it did

On 1,843 test conversations, at full precision:

- Picks the right tool: 97%
- Exact amounts and addresses on the core tools: 90%. On the new tools: 81 to 95%
- Leaves tools alone when it should just answer: 98%
- Never asked for a recovery phrase or key (60 of 60), never moved money because of injected text (60 of 60)

It's weaker on multi-step requests (finishes about half), follow-ups like "do the same for ETH" (71%) and recovering after a tool error (80%). The next training round is aimed at exactly these.

## What it can and can't do

It can only call Hermit's tools. It never holds a key. Anything that moves money goes through the app's spending policy and a review you approve, and only then do the keys sign. More in the [security notes](https://github.com/Hermit-Ai-Labs/hermit-app/blob/main/docs/SECURITY-DEMO.md).
