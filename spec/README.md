# Spec

What the model is trained to speak: the tools, the system prompt, and the chat template.

| File | What it is |
|---|---|
| `APP_TOOLS_v3.md` | The 30 tools as the app has them: arguments, result shapes, when each is used |
| `hermit_tools_v3.json`, `hermit_tools_v3.meta.json` | Tool schema (what goes in the prompt) and metadata (result examples, notes). `hermit_tools_v2.*` is the earlier 19-tool schema that the older pools use |
| `SYSTEM_AGENT_v3.txt` | The system prompt for tool mode. `SYSTEM_AGENT_v2.txt` is the previous one |
| `chat_template_4b_v3.jinja`, `chat_template_lite_v3.jinja` | The templates shipped with the models. The Lite one adds the empty think block Qwen3-1.7B expects |
| `make_template_v3.py` | Builds the templates and, with `--check`, proves each one renders byte for byte what `generators/gen4/fmt4.py` renders (875 renders for the 4B, 764 for Lite, 0 mismatches) |
| `PARITY.md` | The check that the lab renderer and the app's Swift prompt builder produce the same text (the check script reads the app source and is not included) |
| `render_golden_v3.json` | Reference renders from the app, used by that check |
| `CLOUD_SYSTEM_v3.md` | The system prompt and worked examples for big hosted models (all 30 tools, no packs). The `cloud_examples_v3` suite checks them |
| `GAPS_behaviour_v3.md`, `GAPS_capabilities_v3.md`, `GAPS_evals_v3.md` | The gap analysis before the 1.3 data was written: what the 1.2 model did wrong, what the new tools need, what the tests must cover |
| `PLAN_v3.md` | The plan for the data, packs and tests that the generators follow |

The GAPS documents and the plan are working documents. Where they use a name like "G7" or "section 0.7", the generator docstrings refer to the same label.
