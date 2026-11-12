#!/bin/bash
# One-shot Hermit 1.3 GPU run on sft5 (tools v3, tool packs). One box per model, both boxes in parallel, same package:
#   MODE=4b   nohup bash run_remote4.sh > run.log 2>&1 &   # A100 80GB: 4B train -> eval -> control eval of the 1.2b adapter -> ALL_DONE
#   MODE=lite nohup bash run_remote4.sh > run.log 2>&1 &   # A6000 48GB: Lite train -> eval -> ALL_DONE
# Same environment setup and STATUS markers as run_remote3.sh (4B markers, LITE_* markers for Lite). Each output is packed into
# ~/out_*.tgz as soon as its stage ends. Any abort still ends in an ALL_DONE line (with the exit code), so the poller collects and
# stops the box at once instead of waiting for the time cap. Options: SKIP_CONTROL=1 (4B box: no control eval),
# CAP_END=<unix time the watchdog fires> (evals stop 15 min before it; the control eval is skipped with under 25 min left),
# CONTROL_LIMIT (default 40 items per suite for the 1.2b control; the candidate uses 160),
# EPOCHS (default 2.0; the 2026-11-11 L40S rerun of the 4B uses EPOCHS=1.0), MEM_TARGET_GB (memory-probe target; default: 70 on
# an 80 GB card, otherwise 88% of the card's GiB as nvidia-smi reports it: 42 on an A6000, 39 on an L40S).
# Run 6 (2026-11-11, second epoch of the 4B on an L40S), all optional, defaults = the old behaviour:
#   REQUIRE_RESUME=<step>  4B: refuse to train unless runs/h13-4b/last/state.pt exists and holds at least that step (a missing
#                          checkpoint would silently train from the base again). Run 6: 1336 = the one-epoch result, EPOCHS=2.0.
#   EVAL_MARGIN=<seconds>  evals get the deadline CAP_END - EVAL_MARGIN (default 900; no batch starts in its last 3 min).
#   EVAL_PLAN="suites[@chunk[@budget_s]];..."  the candidate eval as several eval_hermit4.py calls with the same tag, in this
#                          order (a rerun with the same tag keeps what is finished): each group is a --suites list, optionally
#                          its own --chunk and a time budget in seconds (what a budget cuts is picked up by a later "all").
set -eo pipefail
MODE=${MODE:-}
case "$MODE" in 4b|lite) ;; *) echo "usage: MODE=4b|lite nohup bash run_remote4.sh > run.log 2>&1 &"; exit 2 ;; esac
cd ~/hermit
S=~/hermit/STATUS
[ -f $S ] && mv $S $S.prev.$(date +%s)        # a rerun starts a fresh STATUS (the poller stops at the first ALL_DONE it sees)
mark() { echo "$(date -u +%H:%M:%S) $*" | tee -a $S; }
STAGE=start
trap 'rc=$?; grep -q "ALL_DONE" $S 2>/dev/null || mark "ALL_DONE aborted (exit $rc in stage $STAGE, see run.log)"' EXIT
mark "START MODE=$MODE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a $S
GMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)      # MiB
CUDA_V=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
MAJ=${CUDA_V%%.*}; MIN=${CUDA_V##*.}
if   [ "$MAJ" -ge 13 ] || { [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; }; then IDX=cu128
elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then IDX=cu126
elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 4 ]; then IDX=cu124
else IDX=cu121; fi
mark "driver CUDA $CUDA_V -> torch index $IDX; GPU memory ${GMEM} MiB"

# ---- env (uv: fast, brings its own python)
STAGE=env
if ! command -v uv >/dev/null 2>&1 && [ ! -x ~/.local/bin/uv ]; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true; fi
export PATH=$HOME/.local/bin:$PATH
[ -x ~/venv/bin/python ] || uv venv -q --python 3.11 ~/venv
. ~/venv/bin/activate
uv pip install -q "torch==2.15.0" --index-url https://download.pytorch.org/whl/$IDX || uv pip install -q torch --index-url https://download.pytorch.org/whl/$IDX   # 2.15.0 = the CPU smoke runs' torch
uv pip install -q "transformers==5.22.2" "peft==0.21.2" "accelerate==1.15.0" "huggingface_hub==2.2.0" safetensors jinja2 pycryptodome
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" | tee -a $S
mark ENV_OK

cd ~/hermit/train
export HERMIT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fetch() {   # repo local_dir (public models, no token)
  python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
}
# micro-batch limits by GPU memory; the step plan (16 rows per step) never depends on them. The memory probe in sft_gpu4.py
# runs the heaviest micro-batches first and shrinks these until the peak is under the target; an OOM later resumes from
# `last` with 2/3 of them (same plan, same step).
limits() {  # family -> MMT MMS MMC TGT
  local tgt
  if [ "$GMEM" -ge 75000 ]; then tgt=70; else tgt=$((GMEM * 88 / 102400)); fi
  tgt=${MEM_TARGET_GB:-$tgt}
  if [ "$GMEM" -ge 75000 ]; then
    [ "$1" = 4b ] && echo "2400 16 36000 $tgt" || echo "3200 16 64000 $tgt"
  else
    [ "$1" = 4b ] && echo "1400 16 20000 $tgt" || echo "2400 16 48000 $tgt"
  fi
}
train() {   # family base out lr rank alpha log
  local fam=$1 base=$2 out=$3 lr=$4 rank=$5 alpha=$6 lg=$7 try
  read MMT MMS MMC TGT <<< "$(limits $fam)"
  mark "${PFX}TRAIN_ARGS epochs ${EPOCHS:-2.0}, micro limits $MMT/$MMS/$MMC, memory-probe target $TGT GB"
  for try in 1 2 3 4; do
    python train/sft_gpu4.py --base $base --family $fam --train data5/sft5_train.jsonl --val data5/sft5_val.jsonl \
      --out $out --epochs ${EPOCHS:-2.0} --lr $lr --rank $rank --alpha $alpha --seqs-per-step 16 \
      --max-micro-tokens $MMT --max-micro-seqs $MMS --max-micro-ctx $MMC --mem-target-gb $TGT --probe 3 \
      --warmup 30 --save-every 25 --eval-every 300 --threads 8 >> $lg 2>&1 && return 0
    mark "${PFX}TRAIN_FAILED try $try, resuming from the last checkpoint with smaller micro-batches"
    MMT=$((MMT * 2 / 3)); MMS=$((MMS * 2 / 3 > 2 ? MMS * 2 / 3 : 2)); MMC=$((MMC * 2 / 3))
  done
  return 1
}
SU=all
# CAP_END (unix time the box's watchdog fires, passed by the controller) -> every eval stops starting new batch-rounds 15 min
# before it, writes what finished (transcripts are kept per item) and exits 0, so packing, copying and terminating finish first.
EM=${EVAL_MARGIN:-900}
DL=""; [ -n "$CAP_END" ] && DL="--deadline $((CAP_END - EM))"
left() { [ -n "$CAP_END" ] && echo $((CAP_END - EM - $(date +%s))) || echo 999999; }
ev() {      # tag base adapter family [limit per suite, default 160] [plan: "suites[@chunk[@budget_s]];..." instead of $SU]
  local plan=${6:-} g su ck bud dl n=0 i=0 now
  if [ -z "$plan" ]; then
    python eval/eval_hermit4.py --base $2 --adapter $3 --family $4 --tag $1 --suites $SU --bs 32 --threads 8 --limit ${5:-160} $DL > eval_$1.log 2>&1 || mark "EVAL_FAILED $1"
  else
    local IFS=';'; set -f; local groups=($plan); set +f; unset IFS
    n=${#groups[@]}
    for g in "${groups[@]}"; do
      i=$((i + 1)); IFS=@ read -r su ck bud <<< "$g"
      [ "$(left)" -lt 200 ] && { mark "EVAL_STAGE $i/$n $su not started (deadline)"; break; }
      dl=$DL; now=$(date +%s)
      if [ -n "$bud" ]; then                                  # the eval starts no batch in the last 180 s before its deadline
        dl="--deadline $((now + bud + 180))"
        [ -n "$CAP_END" ] && [ $((now + bud + 180)) -gt $((CAP_END - EM)) ] && dl=$DL
      fi
      mark "EVAL_STAGE $i/$n $su${ck:+ chunk $ck}${bud:+ budget ${bud}s}"
      echo "==== stage $i/$n $(date -u +%H:%M:%S) suites $su" >> eval_$1.log
      python eval/eval_hermit4.py --base $2 --adapter $3 --family $4 --tag $1 --suites "$su" --bs 32 --threads 8 --limit ${5:-160} ${ck:+--chunk $ck} $dl >> eval_$1.log 2>&1 || mark "EVAL_FAILED $1 stage $i ($su)"
      tar czf ~/out_eval_$1.tgz.tmp eval5/results/$1 eval_$1.log 2>/dev/null && mv -f ~/out_eval_$1.tgz.tmp ~/out_eval_$1.tgz
    done
  fi
  tar czf ~/out_eval_$1.tgz.tmp eval5/results/$1 eval_$1.log 2>/dev/null && mv -f ~/out_eval_$1.tgz.tmp ~/out_eval_$1.tgz || true
  mark "EVAL_DONE $1"
}

if [ "$MODE" = 4b ]; then
  # ================================================================================== Hermit 1.3 (Qwen3-4B-Instruct-2507)
  PFX=""
  STAGE=base
  BASE=$( [ "$(id -u)" = 0 ] && echo /root/base4b || echo ~/base4b )
  fetch Qwen/Qwen3-4B-Instruct-2507 $BASE
  mark BASE_OK
  # offline checks (no GPU): eval oracle + mutations + prompt hashes for the 4B; template == fmt4 render; sft_gpu4 encode == fmt4
  # on 20 random rows. Never block the run.
  STAGE=checks
  timeout 1200 python eval/eval_hermit4.py --dry-run --family 4b > dryrun.log 2>&1 && mark DRYRUN_OK || mark "DRYRUN_FAILED (see dryrun.log), continuing"
  { python spec/make_template_v3.py --check && python train/sft_gpu4.py --base $BASE --family 4b --train data5/sft5_train.jsonl \
      --out runs/render_check_4b --render-check 20; } > template.log 2>&1 && mark RENDER_OK || mark "RENDER_FAILED (see template.log), continuing"
  # train from the BASE: same LoRA recipe as h12-4b-gpu2 (r16/a32, attn+MLP, lr 1.5e-4, 2 epochs, 16 rows per step, warmup 30)
  STAGE=train
  if [ -n "$REQUIRE_RESUME" ]; then      # run 6: continue the saved one-epoch run; never start again from the base by accident
    python - "$REQUIRE_RESUME" <<'PY' && mark "RESUME_OK (checkpoint at step >= $REQUIRE_RESUME)" || { mark "RESUME_MISSING: runs/h13-4b/last is not a checkpoint at step >= $REQUIRE_RESUME, not training"; exit 3; }
import os, sys, torch
d = "runs/h13-4b/last"
assert os.path.getsize(d + "/adapter_model.safetensors") > 10 ** 8 and os.path.exists(d + "/adapter_config.json")
st = torch.load(d + "/state.pt", weights_only=False, map_location="cpu")
print("checkpoint step", st["step"], "scheduler step", st["sched"]["last_epoch"]); sys.exit(0 if st["step"] >= int(sys.argv[1]) else 1)
PY
  fi
  train 4b $BASE runs/h13-4b 1.5e-4 16 32 runs_train.log || mark "TRAIN_GAVE_UP after 4 tries"
  test -f runs/h13-4b/final/adapter_model.safetensors
  mark TRAIN_DONE
  tar czf ~/out_adapter4.tgz runs/h13-4b/final/adapter_model.safetensors runs/h13-4b/final/adapter_config.json runs/h13-4b/train.log runs_train.log
  mark ADAPTER_PACKED
  # evals on eval5 (36 suites, the item's pack, 4B family), --limit 160 per suite; candidate first, then the 1.2b adapter as control
  STAGE=eval
  ev h13-4b-e5 $BASE runs/h13-4b/final 4b 160 "$EVAL_PLAN"
  if [ -n "$SKIP_CONTROL" ]; then mark CONTROL_SKIPPED
  elif [ "$(left)" -lt 1500 ]; then mark "CONTROL_SKIPPED (under 25 min left before the cap)"
  else ev h12-4b-gpu2-e5 $BASE runs/h12-4b-gpu2/final 4b ${CONTROL_LIMIT:-40}; fi    # control: the first 40 items per suite (paired with the candidate's first 40)
  mark FOURB_DONE
else
  # ================================================================================== Hermit 1.3 Lite (Qwen3-1.7B, thinking family)
  # The app renders an empty <think>\n\n</think>\n\n after every assistant prefix; fmt4.render(..., "lite") puts it in the
  # non-trained prefix, so the model learns to answer right after it. Lite trains on the family_ok rows with their Lite pack.
  PFX="LITE_"
  STAGE=base
  BASEL=$( [ "$(id -u)" = 0 ] && echo /root/baselite || echo ~/baselite )
  fetch Qwen/Qwen3-1.7B $BASEL
  mark LITE_BASE_OK
  STAGE=checks
  timeout 1200 python eval/eval_hermit4.py --dry-run --family lite > dryrun_lite.log 2>&1 && mark LITE_DRYRUN_OK || mark "LITE_DRYRUN_FAILED (see dryrun_lite.log), continuing"
  { python spec/make_template_v3.py --check && python train/sft_gpu4.py --base $BASEL --family lite --train data5/sft5_train.jsonl \
      --out runs/render_check_lite --render-check 20; } > template_lite.log 2>&1 && mark LITE_RENDER_OK || mark "LITE_RENDER_FAILED (see template_lite.log), continuing"
  # recipe scaled for 1.7B (as LAUNCH3): LoRA r32/a64 on attn+MLP, lr 2e-4, 2 epochs, 16 rows per step, warmup 30, max-len 6144
  STAGE=train
  train lite $BASEL runs/h13-lite 2e-4 32 64 runs_train_lite.log || mark "LITE_TRAIN_GAVE_UP after 4 tries"
  test -f runs/h13-lite/final/adapter_model.safetensors
  mark LITE_TRAIN_DONE
  tar czf ~/out_adapter_lite.tgz runs/h13-lite/final/adapter_model.safetensors runs/h13-lite/final/adapter_config.json runs/h13-lite/train.log runs_train_lite.log
  mark LITE_ADAPTER_PACKED
  STAGE=eval
  ev h13-lite-e5 $BASEL runs/h13-lite/final lite
fi
STAGE=done
mark ALL_DONE
