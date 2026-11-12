#!/bin/bash
# final.sh <convert args...>: the chosen format as export/Hermit-1.3-Lite-mlx-final, template handling as gpu/finish_local4_lite.sh
set -e
T=train; V=venv/bin/python; EXPORT=$T/export
TPL=$T/spec/chat_template_lite_v3.jinja
OUT=$EXPORT/Hermit-1.3-Lite-mlx-final
rm -rf $OUT
nice -n 19 mlxenv/bin/python -m mlx_lm convert --hf-path $EXPORT/Hermit-1.3-Lite-merged -q "$@" --mlx-path $OUT
cp $TPL $OUT/chat_template.jinja
$V -c "import json,sys; c=json.load(open('$OUT/tokenizer_config.json')); assert c.get('chat_template','')==open('$TPL').read() or 'chat_template' not in c, 'stale template in the MLX tokenizer_config'"
for f in chat_template.jinja tokenizer_config.json tokenizer.json generation_config.json README.md; do cmp $OUT/$f $EXPORT/Hermit-1.3-Lite-mlx/$f && echo "same as -mlx: $f"; done
$V - <<PY
import json
a = json.load(open("$OUT/config.json")); b = json.load(open("$EXPORT/Hermit-1.3-Lite-mlx/config.json"))
d = {k: (b.get(k), a.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
print("config.json differs from -mlx only in:", d)
assert set(d) <= {"quantization", "quantization_config"}
t = open("$OUT/chat_template.jinja").read()
assert "assistant\\\\n<think>\\\\n\\\\n</think>" in t and t.count('"name": "') >= 30
PY
du -sb $OUT; du -sh $OUT; echo FINAL_OK
