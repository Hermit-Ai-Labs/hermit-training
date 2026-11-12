# Generators

The programs that wrote the training rows and the held-out test items. Each one is deterministic: a fixed seed gives the same output.

```
generators/
  gen/      hand-written strategy answers and execution-checked code tasks (restraint_strategy, restraint_code)
  gen2/     the first agent generator: simulator, tool formats, calculator conversations, quote questions,
            the teacher-model client and prompts, and the v2 assemblers (assemble2.py, assemble3.py)
  gen3/     14 pools for the 19-tool app, the v4 assembler (assemble4.py), and the eval3 suites
  gen4/     14 pools for the 30-tool app, tool packs (packs.py), the renderer (fmt4.py), the scorer
            (eval_suites4.py), the final assembler (assemble5.py), and the results/gates script (make_results4.py)
  hermit_tools.json   the original tool list that gen2/hermit_fmt.py reads
```

## Which generator wrote which rows

See the table in `data/README.md`. In short: `gen4/<name>.py` wrote the pool called `<name>`; `gen3/<name>.py` the same for its pools; `gen2/build_agent.py` the `core_*` and `restraint_misc` rows; `gen2/build_teacher.py` the `restraint_teacher` rows; `gen/` the strategy and code rows.

## Shared pieces

| File | Role |
|---|---|
| `gen2/hermit_fmt.py` | Number and money formatting, the calculator, random EVM and Solana addresses, name lists |
| `gen2/sim.py`, `gen3/sims3.py`, `gen4/sims4.py` | The simulated wallet world. Tool calls run against it and return the same JSON shapes the app returns |
| `gen4/fmt4.py` | Renders a conversation exactly as the app does (system prompt, tool block for the pack, device word). Training and tests both use it |
| `gen4/packs.py` | The tool packs and the router that picks one from the first user message |
| `gen4/check4.py` | Checks every generated row (arguments present, nothing invented, pack respected) |
| `gen4/eval_suites4.py` | The scorer: every check named in the results tables is defined here |
| `gen4/assemble5.py` | Merges pools into `sft5_train.jsonl` and `sft5_val.jsonl` |
| `gen4/build_eval5.py` | Builds the 36 test suites in `evals/suites/` |

## Running them

The scripts find their neighbours with paths like `<script dir>/../spec`, so they need the original folder layout. `tools/make_lab.py` rebuilds it:

```
python tools/make_lab.py
cd .lab
python gen4/safety_mixed.py --n 200 --eval-n 60
```

Only Python and `pycryptodome` are needed. The generators use no network and no model. The larger ones (`multistep`, `recovery_errors`, `auto_pages`, `followups_stale`) loop until each row passes its own checks and take minutes even for a small count.

`gen2/teacher.py` and `gen2/build_teacher.py` call a local model server (Qwen3-30B-A3B-Instruct-2507 behind an OpenAI-style endpoint on localhost). They are here so the prompts and filters can be read. They cannot run without such a server.

## What reproduces and what does not

- `gen4/safety_mixed.py` run again with its defaults gave 200 rows equal, message for message, to the 200 `safety_mixed` rows in `data/`, and test items equal to the shipped suite.
- `gen4/assemble5.py` and `gen3/assemble4.py` need earlier training sets and earlier test suites that are not shipped. Read them for the exact filtering rules.
- The teacher rows depend on the teacher model's outputs, which are cached in the original working directory and not shipped. The rows are in `data/`.
- A short hand-written set of voice and honest-limits examples (43 rows) has no generator here. Its rows are in `data/`.
