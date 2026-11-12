#!/bin/bash
# Hermit 1.3 (4B) after its GPU box: unpack back4_4b/, results table + gates (eval5/results_1.3.md, candidate vs the 1.2b control),
# merge the adapter, attach the v3 chat template (30 tools, the app's pack, device word), MLX 4-bit.
# Writes eval5/results/{h13-4b-e5,h12-4b-gpu2-e5}, eval5/results_1.3.md, export/Hermit-1.3-4B-{adapter,merged,mlx}.
# Never touches runs/ (the adapter is kept in export/Hermit-1.3-4B-adapter and in back4_4b/out_adapter4.tgz).
# Overrides (tests): T, EXPORT, ADAPTER, SKIP_UNPACK=1, SKIP_RESULTS=1, SKIP_MLX=1.
set -e
G=gpu; T=${T:-train}; V=venv/bin/python
EXPORT=${EXPORT:-$T/export}; ADAPTER=${ADAPTER:-$EXPORT/Hermit-1.3-4B-adapter}
TPL=$T/spec/chat_template_4b_v3.jinja        # checked against fmt4.render_text by spec/make_template_v3.py
if [ -z "$SKIP_UNPACK" ]; then
  rm -rf $G/unpack4 && mkdir -p $G/unpack4 && cd $G/unpack4
  for f in $G/back4_4b/out_*.tgz; do [ "$(basename $f)" = out_partial.tgz ] && continue; tar xzf $f; done
  if [ ! -f runs/h13-4b/final/adapter_model.safetensors ] && [ -f $G/back4_4b/out_partial.tgz ]; then
    echo "NO FINAL ADAPTER: unpacking out_partial.tgz (logs, eval results so far, the newest 'last' adapter)"; tar xzf $G/back4_4b/out_partial.tgz
  fi
  mkdir -p $ADAPTER
  cp runs/h13-4b/final/adapter_model.safetensors runs/h13-4b/final/adapter_config.json $ADAPTER/
  cp runs/h13-4b/train.log $ADAPTER/ 2>/dev/null || true
  for t in h13-4b-e5 h12-4b-gpu2-e5; do [ -d eval5/results/$t ] && rm -rf $T/eval5/results/$t && mkdir -p $T/eval5/results && cp -r eval5/results/$t $T/eval5/results/; done
fi
cd $T
VERDICT=SKIPPED
if [ -z "$SKIP_RESULTS" ]; then
  CTRL=""; [ -d eval5/results/h12-4b-gpu2-e5 ] && CTRL="--control h12-4b-gpu2-e5"
  $V gen4/make_results4.py h13-4b-e5 --family 4b $CTRL && VERDICT=PASS || VERDICT=FAIL
fi
# merge anyway so the model can be inspected
nice -n 19 $V export/merge_export2.py --base ../base/Qwen3-4B-Instruct-2507 --adapter $ADAPTER --family 4b --out $EXPORT/Hermit-1.3-4B-merged
cp $TPL $EXPORT/Hermit-1.3-4B-merged/chat_template.jinja
TPL=$TPL OUT=$EXPORT/Hermit-1.3-4B-merged $V - <<'PY'
import json, os
p = os.environ["OUT"] + "/tokenizer_config.json"; c = json.load(open(p)); t = open(os.environ["TPL"]).read()
c["chat_template"] = t; json.dump(c, open(p, "w"), indent=1, ensure_ascii=False)
assert "assistant\\n<think>\\n\\n</think>" not in t                 # 4B = instruct family (no think block)
assert t.count('"name": "') >= 30 and "prepare_sell" in t and "get_address" in t
print("template ok:", len(t), "chars, 30 v3 tools, pack chosen by the passed tools, device word from the system prompt")
PY
if [ -z "$SKIP_MLX" ]; then
  rm -rf $EXPORT/Hermit-1.3-4B-mlx
  nice -n 19 mlxenv/bin/python -m mlx_lm convert --hf-path $EXPORT/Hermit-1.3-4B-merged -q --q-bits 4 --q-group-size 64 --mlx-path $EXPORT/Hermit-1.3-4B-mlx
  cp $TPL $EXPORT/Hermit-1.3-4B-mlx/chat_template.jinja
  $V -c "import json,sys; c=json.load(open('$EXPORT/Hermit-1.3-4B-mlx/tokenizer_config.json')); assert c.get('chat_template','')==open('$TPL').read() or 'chat_template' not in c, 'stale template in the MLX tokenizer_config'"
  du -sh $EXPORT/Hermit-1.3-4B-mlx
fi
echo "FINISH_LOCAL4 OK verdict=$VERDICT (results: eval5/results_1.3.md). Mixed-precision export and the Mac agent-check come after a PASS (LAUNCH4.md section 6)."
