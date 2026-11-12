#!/bin/bash
# Hermit 1.3 Lite (Qwen3-1.7B) after its GPU box: unpack back4_lite/, Lite gates (eval5/results_1.3_lite.md, the 4B candidate as a
# reference column when finish_local4.sh already ran), merge the adapter, attach the Lite v3 chat template (30 tools, the app's pack,
# empty think block in every assistant prefix and in the generation prompt), MLX 4-bit. Independent of finish_local4.sh.
# Writes eval5/results/h13-lite-e5, eval5/results_1.3_lite.md, export/Hermit-1.3-Lite-{adapter,merged,mlx}. Never touches runs/.
# Overrides (tests): T, EXPORT, ADAPTER, SKIP_UNPACK=1, SKIP_RESULTS=1, SKIP_MLX=1.
set -e
G=gpu; T=${T:-train}; V=venv/bin/python
EXPORT=${EXPORT:-$T/export}; ADAPTER=${ADAPTER:-$EXPORT/Hermit-1.3-Lite-adapter}
TPL=$T/spec/chat_template_lite_v3.jinja      # checked against fmt4.render_text(..., "lite") by spec/make_template_v3.py
if [ -z "$SKIP_UNPACK" ]; then
  rm -rf $G/unpack4_lite && mkdir -p $G/unpack4_lite && cd $G/unpack4_lite
  for f in $G/back4_lite/out_*.tgz; do [ "$(basename $f)" = out_partial.tgz ] && continue; tar xzf $f; done
  if [ ! -f runs/h13-lite/final/adapter_model.safetensors ] && [ -f $G/back4_lite/out_partial.tgz ]; then
    echo "NO FINAL ADAPTER: unpacking out_partial.tgz (logs, eval results so far, the newest 'last' adapter)"; tar xzf $G/back4_lite/out_partial.tgz
  fi
  mkdir -p $ADAPTER
  cp runs/h13-lite/final/adapter_model.safetensors runs/h13-lite/final/adapter_config.json $ADAPTER/
  cp runs/h13-lite/train.log $ADAPTER/ 2>/dev/null || true
  [ -d eval5/results/h13-lite-e5 ] && rm -rf $T/eval5/results/h13-lite-e5 && mkdir -p $T/eval5/results && cp -r eval5/results/h13-lite-e5 $T/eval5/results/
fi
cd $T
VERDICT=SKIPPED
if [ -z "$SKIP_RESULTS" ]; then
  REF=""; [ -d eval5/results/h13-4b-e5 ] && REF="--ref h13-4b-e5"
  $V gen4/make_results4.py h13-lite-e5 --family lite $REF && VERDICT=PASS || VERDICT=FAIL
fi
# merge anyway so the model can be inspected (tied embeddings: lm_head shares embed_tokens, merge_and_unload handles it)
nice -n 19 $V export/merge_export2.py --base ../base/Qwen3-1.7B --adapter $ADAPTER --family lite --out $EXPORT/Hermit-1.3-Lite-merged
cp $TPL $EXPORT/Hermit-1.3-Lite-merged/chat_template.jinja
TPL=$TPL OUT=$EXPORT/Hermit-1.3-Lite-merged $V - <<'PY'
import json, os
p = os.environ["OUT"] + "/tokenizer_config.json"; c = json.load(open(p)); t = open(os.environ["TPL"]).read()
c["chat_template"] = t; json.dump(c, open(p, "w"), indent=1, ensure_ascii=False)
assert "assistant\\n<think>\\n\\n</think>" in t      # what LocalAgent.family() looks for (the repo name should also contain "lite")
assert t.count('"name": "') >= 30 and "prepare_sell" in t and "get_address" in t
print("template ok:", len(t), "chars, 30 v3 tools, empty think block in the assistant prefix and the generation prompt")
PY
if [ -z "$SKIP_MLX" ]; then
  rm -rf $EXPORT/Hermit-1.3-Lite-mlx
  nice -n 19 mlxenv/bin/python -m mlx_lm convert --hf-path $EXPORT/Hermit-1.3-Lite-merged -q --q-bits 4 --q-group-size 64 --mlx-path $EXPORT/Hermit-1.3-Lite-mlx
  cp $TPL $EXPORT/Hermit-1.3-Lite-mlx/chat_template.jinja
  $V -c "import json,sys; c=json.load(open('$EXPORT/Hermit-1.3-Lite-mlx/tokenizer_config.json')); assert c.get('chat_template','')==open('$TPL').read() or 'chat_template' not in c, 'stale template in the MLX tokenizer_config'"
  du -sh $EXPORT/Hermit-1.3-Lite-mlx
fi
echo "FINISH_LOCAL4_LITE OK verdict=$VERDICT (results: eval5/results_1.3_lite.md). Upload under a repo name containing 'lite' (the app picks the thinking family from it)."
