# Training

LoRA fine-tuning of Qwen3-4B-Instruct-2507 (Hermit 1.3) and Qwen3-1.7B (Hermit 1.3 Lite) on `data/sft5_train.jsonl.gz`.

| File | What it is |
|---|---|
| `sft_gpu4.py` | The trainer (PyTorch, Transformers, PEFT). The top docstring explains each design choice |
| `run_remote4.sh` | The run script used on the rented machines: sets up the environment, downloads the base model, checks the prompt rendering, trains, runs the tests, packs the outputs |
| `adapter_configs/` | The LoRA settings of the two adapters as saved |
| `logs/train_4b.log.gz`, `logs/train_lite.log.gz` | The full training logs: every step's loss, learning rate and memory, and every validation loss |

## Settings

| | Hermit 1.3 (4B) | Hermit 1.3 Lite |
|---|---|---|
| Base | Qwen/Qwen3-4B-Instruct-2507 | Qwen/Qwen3-1.7B |
| LoRA | r 16, alpha 32, dropout 0.05 | r 32, alpha 64, dropout 0.05 |
| Target modules | q, k, v, o, gate, up, down projections | same |
| Trainable parameters | 33,030,144 | 34,865,152 |
| Learning rate | 1.5e-4 | 2e-4 |
| Schedule | 30 warm-up steps, cosine down to 10% of peak | same |
| Optimiser | AdamW, betas (0.9, 0.99), weight decay 0, gradient clip 1.0 | same |
| Precision | bf16 base, fp32 LoRA weights | same |
| Rows per step | 16 | 16 |
| Epochs | 2 | 2 |
| Rows | 21,361 | 18,917 (rows with `lite` in `meta.family_ok`) |
| Steps | 2,672 (1,336 per epoch) | 2,366 (1,183 per epoch) |
| Max length | 8,192 tokens (longest row 7,641) | 6,144 tokens (longest row 4,713) |
| Trained tokens per epoch | 2,962,513 | 2,277,490 |
| Seed | 0 | 0 |
| GPU | NVIDIA L40S, 46 GB | NVIDIA RTX A6000, 48 GB |
| Validation loss | 0.1742 after epoch 1, 0.1527 at the end | 0.3598 at step 300, 0.2015 at the end |

Loss is only on the assistant's tokens (including its tool calls). The system prompt, the tool block, user turns and tool results are context.

The trainer encodes the system prompt and tool block once per tool pack and reuses it (a prefix cache), with gradients flowing through everything after it. A step always holds exactly 16 rows, whatever the micro-batch size, so a small GPU and a big one train on the same steps. Before training it runs the heaviest micro-batches and shrinks the limits until the peak memory fits the card.

## Commands

The data goes in `data5/`, the base model in `~/base4b` or `~/baselite`, and the repo's `generators/gen4`, `spec/`, `evals/` and `training/` in the layout `tools/make_lab.py` builds. `run_remote4.sh` expects that layout under `~/hermit/train`.

### Hermit 1.3 (4B)

Pass 1, one epoch:

```
MODE=4b SKIP_CONTROL=1 EPOCHS=1.0 MEM_TARGET_GB=46 bash run_remote4.sh
```

which runs

```
python train/sft_gpu4.py --base ~/base4b --family 4b \
  --train data5/sft5_train.jsonl --val data5/sft5_val.jsonl --out runs/h13-4b \
  --epochs 1.0 --lr 1.5e-4 --rank 16 --alpha 32 --seqs-per-step 16 \
  --max-micro-tokens 1400 --max-micro-seqs 16 --max-micro-ctx 20000 \
  --mem-target-gb 46 --probe 3 --warmup 30 --save-every 25 --eval-every 300 --threads 8
```

Pass 2, on a fresh machine, resuming the checkpoint that pass 1 left in `runs/h13-4b/last` (adapter, optimiser and scheduler state at step 1,336):

```
MODE=4b SKIP_CONTROL=1 EPOCHS=2.0 MEM_TARGET_GB=46 REQUIRE_RESUME=1336 bash run_remote4.sh
```

The same command with `--epochs 2.0`. The trainer finds the checkpoint, checks it belongs to the same step plan, and continues at step 1,337. `REQUIRE_RESUME` makes the script refuse to start from the base model by accident.

About the learning rate: pass 1 ran a one-epoch cosine, so it ended at the floor (1.5e-5). Pass 2 used the two-epoch plan, so at step 1,338 the rate went back up to 8.2e-5 and decayed again to 1.5e-5 at step 2,672. The training log shows both.

Timing: pass 1 took about 70 minutes of training (the first start was stopped and resumed once, at step 175, to fix a memory setting) and pass 2 about 66 minutes. Peak memory was 44.7 GB.

An earlier attempt to train the 4B in one run on an A100 was lost when the rented machine became unreachable before its results were collected, so it is not part of the model. The adapter that shipped comes from the two L40S passes above.

### Hermit 1.3 Lite

One run of two epochs:

```
MODE=lite bash run_remote4.sh
```

which runs

```
python train/sft_gpu4.py --base ~/baselite --family lite \
  --train data5/sft5_train.jsonl --val data5/sft5_val.jsonl --out runs/h13-lite \
  --epochs 2.0 --lr 2e-4 --rank 32 --alpha 64 --seqs-per-step 16 \
  --max-micro-tokens 2400 --max-micro-seqs 16 --max-micro-ctx 48000 \
  --mem-target-gb 42 --probe 3 --warmup 30 --save-every 25 --eval-every 300 --threads 8
```

It took about 1 hour 53 minutes. Lite renders every assistant turn after an empty `<think>` block, as Qwen3-1.7B expects; `fmt4.render(..., "lite")` puts that block in the part of the sequence that is not trained.

### Software

As pinned in `run_remote4.sh`: Python 3.11, `torch` 2.11 (CUDA 12.8 build), `transformers` 5.22.2, `peft` 0.21.2, `accelerate` 1.15.0, `huggingface_hub` 2.2.0.

### Checks before training

The script runs two offline checks first and continues even if they fail (they did not):

- `eval_hermit4.py --dry-run`: the reference answer of every test item passes its checks.
- `spec/make_template_v3.py --check` and `sft_gpu4.py --render-check 20`: the chat template, the renderer and the trainer's encoder agree on 20 random rows, and the trained tokens are exactly the assistant segments.

## After training

The adapter is merged into the base model and converted for MLX. See `../quantisation/README.md`. The test run is described in `../evals/README.md`.
