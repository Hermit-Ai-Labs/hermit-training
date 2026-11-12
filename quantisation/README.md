# Quantisation

From the trained adapter to the files the app downloads.

1. **Merge** the LoRA adapter into the bf16 base model.
2. **Attach the app's chat template** (`spec/chat_template_4b_v3.jinja` or `spec/chat_template_lite_v3.jinja`) and the generation settings (temperature 0.6, top-p 0.9, top-k 20, repetition penalty 1.05).
3. **Convert to MLX** with a fixed bit width and group size 64.

## Commands

Hermit 1.3 (4B):

```
python merge_export2.py --base Qwen3-4B-Instruct-2507 --adapter Hermit-1.3-4B-adapter --family 4b --out Hermit-1.3-4B-merged
cp ../spec/chat_template_4b_v3.jinja Hermit-1.3-4B-merged/chat_template.jinja     # and set "chat_template" in tokenizer_config.json to the same text
python -m mlx_lm convert --hf-path Hermit-1.3-4B-merged -q --q-bits 5 --q-group-size 64 --mlx-path Hermit-1.3-4B-MLX-5bit
cp ../spec/chat_template_4b_v3.jinja Hermit-1.3-4B-MLX-5bit/chat_template.jinja
```

Hermit 1.3 Lite:

```
python merge_export2.py --base Qwen3-1.7B --adapter Hermit-1.3-Lite-adapter --family lite --out Hermit-1.3-Lite-merged
cp ../spec/chat_template_lite_v3.jinja Hermit-1.3-Lite-merged/chat_template.jinja
python -m mlx_lm convert --hf-path Hermit-1.3-Lite-merged -q --q-bits 6 --q-group-size 64 --mlx-path Hermit-1.3-Lite-MLX-6bit
cp ../spec/chat_template_lite_v3.jinja Hermit-1.3-Lite-MLX-6bit/chat_template.jinja
```

`merge_export2.py` first attaches the older 12-tool templates in this folder (`chat_template_4b.jinja`, `chat_template_lite.jinja`) and the second line replaces them with the v3 template. Both steps are in the original scripts too.

The conversions were run on Linux with mlx-lm 0.32.2 (MLX 0.32.4, CPU build). The files are plain safetensors, so they load on the Mac and iPhone unchanged. The 4B comes out at 5.5 bits per weight (about 2.6 GB) and Lite at 6.5 (about 1.4 GB), because group-wise scales count too.

After each conversion the script checks that the tokenizer config holds exactly the v3 template, and that the tokenizer, generation config and everything in `config.json` except the quantisation block are identical to the 4-bit export of the same model.

## Why 5-bit and 6-bit

Both models were first exported at 4-bit (group size 64). Then each format was compared with the full-precision model on 40 prompts taken from the test suites, looking at the first step of the assistant (which tool, with which arguments, or a plain answer).

- **Hermit 1.3 (4B):** 5-bit agreed with the full-precision model on 37 of the 40 first steps. 4-bit agreed on 34.
- **Hermit 1.3 Lite:** full precision got 36 of 40 first steps right. 6-bit got 35. 4-bit got 26.

These are the numbers in the model cards (`../cards/`). A few other formats were converted and compared as well (4-bit with group size 32, a mixed 4-bit and 6-bit export, and for the 4B also 6-bit and 8-bit). The 4B shipped at 5-bit and Lite at 6-bit.

The comparison used the quantised weights converted back to float and run through the same test code on the Linux machine. The planned test of the finished MLX build against full precision on a Mac, over 1,500 items, was not run, and the results tables list it as "n/a".

## Files

| File | What it is |
|---|---|
| `merge_export2.py` | Merges an adapter into the base and writes the files the app expects |
| `chat_template_4b.jinja`, `chat_template_lite.jinja` | Older 12-tool templates that `merge_export2.py` attaches before the v3 template replaces them |
| `original/finish_local4_4b.sh`, `original/finish_local4_lite.sh` | The scripts used after each training run: unpack, build the results table, merge, attach the template, 4-bit MLX export. They refer to the working directory they ran in |
| `original/finish_final_4b.sh`, `original/finish_final_lite.sh` | The step that published the chosen 5-bit and 6-bit builds with the same checks |
