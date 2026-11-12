# Notice

Hermit 1.3 and Hermit 1.3 Lite are fine-tunes of open models. This repo does not contain any model weights.

| Model | Used for | Licence |
|---|---|---|
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Base of Hermit 1.3 | Apache-2.0 |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Base of Hermit 1.3 Lite | Apache-2.0 |
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Wrote the general knowledge answers in `restraint_teacher` (run locally) | Apache-2.0 |

Libraries used by the scripts: PyTorch, Transformers, PEFT and Accelerate (training), MLX and mlx-lm (conversion). Each keeps its own licence.

The published Hermit models carry the Apache-2.0 licence of their base models. Hermit, the Hermit wallet and the Hermit logo are names of Hermit AI Labs. Other product names in the data (Hyperliquid, Aave, Jupiter, NEAR Intents and so on) belong to their owners and appear only to describe what the app connects to.
