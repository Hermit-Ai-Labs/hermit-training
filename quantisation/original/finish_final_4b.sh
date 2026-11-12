#!/bin/bash
# quant_check_4b: publish the chosen MLX format as export/Hermit-1.3-4B-mlx-final.
# variants/<name> was made by the same command as finish_local4.sh (mlx_lm convert of Hermit-1.3-4B-merged, only the bits differ);
# the template step and its check below are finish_local4.sh's. Usage: finish_final.sh q5
set -e
N=$1; H=gpu/quant_check_4b; T=train; V=venv/bin/python
EXPORT=$T/export; TPL=$T/spec/chat_template_4b_v3.jinja; OUT=$EXPORT/Hermit-1.3-4B-mlx-final; CUR=$EXPORT/Hermit-1.3-4B-mlx
[ -f $H/variants/$N/model.safetensors ] || { echo "no variant $N"; exit 1; }
rm -rf $OUT.tmp && cp -r $H/variants/$N $OUT.tmp
cp $TPL $OUT.tmp/chat_template.jinja
$V -c "import json,sys; c=json.load(open('$OUT.tmp/tokenizer_config.json')); assert c.get('chat_template','')==open('$TPL').read() or 'chat_template' not in c, 'stale template in the MLX tokenizer_config'"
# same template, tokenizer and (apart from the quantization blocks) config as the current 4-bit export
cmp $OUT.tmp/chat_template.jinja $CUR/chat_template.jinja
cmp $OUT.tmp/tokenizer.json $CUR/tokenizer.json
cmp $OUT.tmp/tokenizer_config.json $CUR/tokenizer_config.json
cmp $OUT.tmp/generation_config.json $CUR/generation_config.json
$V - <<PY
import json
a = json.load(open("$OUT.tmp/config.json")); b = json.load(open("$CUR/config.json"))
q = a["quantization"]
assert a["quantization_config"] == q
assert {k: v for k, v in a.items() if not k.startswith("quantization")} == {k: v for k, v in b.items() if not k.startswith("quantization")}
flat = {k: v for k, v in q.items() if not isinstance(v, dict)}
per_layer = {k: v for k, v in q.items() if isinstance(v, dict)}
print("quantization block:", flat, "| per-layer entries:", len(per_layer))
assert set(flat) == {"group_size", "bits", "mode"} and flat["mode"] == "affine"
PY
rm -rf $OUT && mv $OUT.tmp $OUT
du -sh $OUT; ls -la $OUT
