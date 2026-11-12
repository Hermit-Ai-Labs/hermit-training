# Hermit 1.3 training

This repo shows how the two on-device models behind the Hermit wallet were made: **Hermit 1.3** (4B) and **Hermit 1.3 Lite** (1.7B). It holds the training data, the programs that wrote it, the test suites and scorer, the real results, and the exact training and quantisation steps.

Hermit is a wallet for iPhone and Mac with an AI assistant that runs on the device. The assistant never holds a key. It calls a fixed set of tools (30 in this version), and anything that moves money becomes a review the user approves with Face ID or Touch ID.

- App: https://github.com/Hermit-Ai-Labs/hermit-app
- Models: [Hermit-1.3-4B-MLX-5bit](https://huggingface.co/aiquantgod/Hermit-1.3-4B-MLX-5bit), [Hermit-1.3-Lite-MLX-6bit](https://huggingface.co/aiquantgod/Hermit-1.3-Lite-MLX-6bit)
- Docs: https://docs.hermit.my/dev/model-and-tools/

## The numbers

| | Hermit 1.3 | Hermit 1.3 Lite |
|---|---|---|
| Base model | Qwen3-4B-Instruct-2507 (Apache-2.0) | Qwen3-1.7B (Apache-2.0) |
| Method | LoRA, r 16, alpha 32, attention + MLP | LoRA, r 32, alpha 64, attention + MLP |
| Trainable parameters | 33,030,144 | 34,865,152 |
| Training rows | 21,361 | 18,917 (the rows that fit a Lite prompt) |
| Validation rows | 414 | 367 |
| Passes | 2 epochs, 2,672 steps | 2 epochs, 2,366 steps |
| GPU | NVIDIA L40S 48 GB | NVIDIA RTX A6000 48 GB |
| Shipped format | MLX 5-bit, about 2.6 GB | MLX 6-bit, about 1.4 GB |
| Test set | 3,700 held-out items in 36 suites | same |

Both models learn the same 30 tools. The app sends only a small "pack" of tools with each request (none in plain chat, otherwise 12 to 30), so the prompt stays short. The data teaches the model to work with packs.

## What is in the repo

| Folder | What it holds |
|---|---|
| `data/` | The 21,361 training and 414 validation conversations, statistics, a 20-conversation sample, and a data card |
| `generators/` | The programs that wrote the data, plus the assembler that merged the pools |
| `spec/` | The tool schema (v3), the system prompts, the chat templates, the plan and gap documents |
| `evals/` | The 36 test suites, the scorer, the harness that ran the models, and all results (including the Hermit 1.2 and base-model baselines) |
| `training/` | The trainer, the run script, the LoRA settings, and the training logs |
| `quantisation/` | Merge and MLX conversion scripts |
| `cards/` | The two model cards |
| `tools/` | A data loader and a script that rebuilds the layout the generators expect |

Start with `data/README.md` to see what the model was taught, then `evals/README.md` for how it was tested.

## How the pieces fit

1. **Spec.** `spec/APP_TOOLS_v3.md` and `spec/hermit_tools_v3.json` describe the 30 tools exactly as the app has them. `spec/SYSTEM_AGENT_v3.txt` is the system prompt. The chat templates in `spec/` render the same prompt the app builds in Swift, and `spec/make_template_v3.py --check` proves they match.
2. **Simulator.** Every generated conversation runs against a small simulated wallet world (`generators/gen2/sim.py`, `gen3/sims3.py`, `gen4/sims4.py`). Tool results in the data come from the simulator, and the assistant's answers are built from those results. That keeps numbers, addresses and claims consistent.
3. **Generators.** One program per behaviour (`generators/gen4/*.py`, `gen3/*.py`). Each writes training rows and held-out test items from different seeds.
4. **Assembly.** `generators/gen4/assemble5.py` merged the pools, dropped rows that were wrong for tools v3, removed any training row that matched a test item, de-duplicated, chose a tool pack for every row, and split train and validation. `data/STATS.md` is its output.
5. **Training.** `training/sft_gpu4.py` fine-tunes with LoRA. Loss is only on the assistant's tokens.
6. **Tests.** `generators/gen4/eval_suites4.py` scores a conversation against the checks of its suite. `evals/harness/eval_hermit4.py` runs a model through every suite with a live simulator.
7. **Export.** The adapter is merged into the base, the app's chat template is attached, and `mlx_lm convert` makes the 5-bit and 6-bit builds.

## Where the data came from

The 21,361 training rows come from three generations of generators. The full table is in `data/README.md`.

| Part | Rows | Made by |
|---|---:|---|
| Pools for the v3 tools (14 behaviours, such as spot Solana buys, orders and stops, contacts and payments, multi-step chains, error recovery, follow-ups) | 6,853 | `generators/gen4/` |
| Pools from the 19-tool generation (Aave, multisend, send without a network, route rounding, honesty, file questions, and others), re-rendered for v3 | 5,702 | `generators/gen3/` |
| Calculator, price, balance and trade conversations, plus send-scam warnings and quote questions | 6,664 | `generators/gen2/build_agent.py`, `build_quote.py` |
| General knowledge answers written by an open teacher model (Qwen3-30B-A3B-Instruct-2507) | 1,749 | `generators/gen2/build_teacher.py` |
| Hand-written strategy answers (261), execution-checked code tasks (89), voice and honest-limits examples (43) | 393 | `generators/gen/`, plus a short hand-written set |

Every address in the data is random (seeded `random`), every number comes from code or a simulated tool result, and no real keys or seed phrases are used. Details are in `data/README.md`.

## Regenerate the data

Regeneration is partial, and that matters.

- The **gen4 and gen3 pools** are deterministic. A fixed seed gives the same rows. I checked this on `safety_mixed`: running it again gave 200 training rows that match the 200 rows in `data/` message for message, and 60 test items identical to the shipped suite.
- The **assembler** (`assemble5.py`) needs the earlier version-2 training set and the older eval suites as inputs. Those are not in this repo. The assembler is included so you can read exactly what was filtered and why, but it will not run from this repo alone.
- The **teacher answers** (1,749 rows) came from a local model server. The prompts and the filter are in `generators/gen2/build_teacher.py`; the cached model outputs are not shipped. The rows themselves are in `data/`.

To run any generator:

```
python tools/make_lab.py                 # builds ./.lab with the folder layout the scripts expect
cd .lab
python gen4/safety_mixed.py --n 200 --eval-n 60      # about a second
```

Needs Python (tested on 3.12) and `pycryptodome` (for checksummed addresses). Output goes to `.lab/data4/` or `.lab/data5/pools/`, and test items to `.lab/eval4/` or `.lab/eval5/`. No network, no GPU. The big generators (`multistep`, `recovery_errors`, `auto_pages` and others) search for rows that pass their own checks and take minutes even for a small count; run them with their default counts and wait.

Read the data:

```
python tools/load_data.py                # prints train 21361 and val 414
zcat data/sft5_train.jsonl.gz | head -n 1
```

## Run the tests

The scorer and the harness are plain Python. The dry run needs no model:

```
python tools/make_lab.py
cd .lab
python eval/eval_hermit4.py --dry-run --family 4b
```

It checks that the reference answer of each of the 3,700 items passes its own checks (oracle 100%) and that deliberately broken answers fail.

To rebuild a results table from the recorded transcripts:

```
cd .lab
python gen4/make_results4.py h13-4b-e5 --family 4b
python gen4/make_results4.py h13-lite-e5 --family lite
```

This reproduces `evals/results/results_1.3_4b_pass1.md` and `evals/results/results_1.3_lite.md` line for line (apart from the base-model path, which was shortened). For the partial pass 2 run, copy `evals/results/h13-4b-e5-pass2-partial` and pass that name.

To test a model of your own you need a GPU or a lot of patience on CPU; see `evals/README.md`.

## How the two passes were trained

The full commands and settings are in `training/README.md`. In short:

- **Hermit 1.3 (4B).** LoRA on every attention and MLP projection, rank 16, alpha 32, dropout 0.05, learning rate 1.5e-4 with 30 warm-up steps and cosine decay to 10%, AdamW, 16 conversations per step, loss on assistant tokens only, bf16 base with fp32 adapter weights. Pass 1 trained one epoch (1,336 steps) on an L40S and stopped with a checkpoint. Pass 2 resumed from that checkpoint on a fresh L40S and ran to step 2,672. Validation loss was 0.1742 after pass 1 and 0.1527 after pass 2.
- **Hermit 1.3 Lite.** The same recipe scaled up: rank 32, alpha 64, learning rate 2e-4, 18,917 rows, two epochs in one run on an A6000. Validation loss went from 0.3598 (step 300) to 0.2015 (step 2,366).

## Quantisation

The merged model is converted with `mlx_lm convert -q --q-bits 5 --q-group-size 64` (4B) and `--q-bits 6 --q-group-size 64` (Lite). We picked these by comparing formats on 40 first steps against the full-precision model. 4-bit lost too much on Lite (26 of 40 first steps right against 36 at full precision; 6-bit gets 35). On the 4B, 5-bit agreed with full precision on 37 of 40 first steps and 4-bit on 34. See `quantisation/README.md`.

## Results

All numbers come from the recorded runs in `evals/results/`. Each suite is capped at 160 items, greedy decoding. Raw transcripts are included, so every number can be recounted.

The pass criteria in `evals/suites/README.md` are strict (many are 100%). Many were missed, and the tables below show it.

### Hermit 1.3 (4B)

Pass 1 was scored on all suites (1,801 agent conversations). The eval after pass 2 ran out of time and covered the older suites plus part of two new ones. Where both exist, pass 2 is better.

| Check | Pass 1 | Pass 2 (partial run) | Lite (after 2 passes) |
|---|---|---|---|
| Claimed something was done without a tool result (no_unbacked_done) | 0 of 1,801 | 1 of 316 | 8 of 1,843 |
| Asked for a recovery phrase or key | 0 of 60 | not run | 0 of 60 |
| Money call from injected text | 1 of 60 | not run | 0 of 60 |
| Exact arguments, spot Solana / orders / contacts / automations / drift fixes | 87 / 78 / 85 / 80 / 86% | not run | 88 / 95 / 90 / 81 / 91% |
| Multi-step chains finished | 62.5% (75 of 120) | 72.5% (87 of 120) | 49.5% (45 of 91) |
| Follow-ups, conversation clean | 51.7% (62 of 120) | 67.3% (33 of 49, partial) | 52.3% (58 of 111) |
| Recovery after a tool error | 78.3% | not run | 79.7% |
| Answers without a tool when it should (400-question set, first 160) | not run | 157 of 160 | 157 of 160 |
| Restraint on the new suites, pooled | 88.2% | 100% (68 of 68, partial) | 91.5% |
| Items with no wrong money call, pooled | 93.6% | 91.5% (partial) | 94.4% |
| Maths with the calculator (strict) | not run | 158 of 160 | 158 of 160 |
| Hard-line refusals (old set) | not run | 6 of 6 | 5 of 6 |
| Multi-turn chats clean | not run | 44 of 50 | 42 of 50 |
| Attached-file questions fully right | 14 of 37 (rest not finished) | not run | 31 of 50 |
| Notes (the user's own `hermit.md` lines) followed | 23 of 50 | not run | 23 of 46 |
| Capability honesty, invented feature claims | 1 of 51 | not run | 3 of 51 |

Plainly: both models are good at the core jobs (right tool, no false "done", no secret requests, calculator maths). They are weak at multi-step chains, follow-ups, attached-file questions and following the user's notes. The Lite model finishes about half of multi-step chains. The 4B's pass 2 helped chains and follow-ups but it was only partly re-scored, and the run had one false "done" claim in the multi-step suite that pass 1 did not.

The gate tables in `evals/results/` show every check with its threshold. 25 of the 4B's pass 1 gates and 20 of Lite's were missed.

### Baselines: Hermit 1.2 and the base model

Older runs of Hermit 1.2 against the untouched base (Qwen3-4B-Instruct-2507) on the earlier version-2 test sets, with the 12-tool prompt the 1.2 app used. These are a different test set from the 1.3 suites, so do not put the numbers side by side. There was no head-to-head of 1.2 and 1.3 on the 1.3 suites, because the planned control run was skipped.

| Check | Base 4B | Hermit 1.2 (first GPU run) | Hermit 1.2b (second run) |
|---|---|---|---|
| Prepares the trade, swap or send when asked | 20 of 41 | 38 of 41 | 41 of 41 |
| Says "prepared" with no tool call | 9 | 2 | 0 |
| Right first tool | 77% | 97% | 100% |
| Exact arguments | 57% | 86% | 88% |
| Maths with the calculator (43-category set) | 40% | 98% | 99% |
| Answers a knowledge question without a tool (160) | 154 | 78 | 159 |
| Quote questions call `get_quote`, swaps call `prepare_swap` | 18 of 20, 2 of 12 | 19 of 20, 6 of 12 | 20 of 20, 12 of 12 |
| Trading-code tasks passed | 4 of 6 | 4 of 6 | 4 of 6 |

The three tables are in `evals/baselines/`.

## Notes on what is not here

- No model weights, adapters or checkpoints. The models are on Hugging Face.
- The 1.3 data is the version-5 set. Earlier versions of the data and of the test sets are not included, only the lineage (see `data/README.md`).
- The human-written "real use" set and the extra hard-refusal and injection suites in the plan were never built. The scorer lists those gates as "not built".

## License

Code: MIT, see `LICENSE`. Data and test suites (`data/`, `evals/suites/`, and the recorded results): CC BY 4.0, see `data/LICENSE`. Base models: Apache-2.0, see `NOTICE.md`.
