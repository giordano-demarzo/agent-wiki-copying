#!/usr/bin/env bash
# Fig. 4a/4b on the local models, one after the other on one GPU: serve the model with vLLM, wait until it is
# ready, run a pilot and the full design of replicate.py, stop the server. Logs in lab/logs/.
#     bash run_replications.sh gemma-4-26b-a4b qwen3.8-27b ministral-3-14b
#     DESIGNS=PR PILOT=0 bash run_replications.sh gpt-oss-20b gemma-4-31b      # other designs, no pilot
set -u
cd "$(dirname "$0")"
mkdir -p logs
source /opt/conda/etc/profile.d/conda.sh && conda activate llm-vllm
COMMON="--port 8011 --max-model-len 32768 --gpu-memory-utilization 0.92 --enable-auto-tool-choice"
DESIGNS=${DESIGNS:-AB}
PILOT=${PILOT:-5}
TEXT_ONLY="--limit-mm-per-prompt {\"image\":0,\"video\":0}"   # the multimodal models read no images

serve_args() {   # the served name -> the model and its vLLM options
  case "$1" in
    gpt-oss-20b)     echo "openai/gpt-oss-20b --tool-call-parser openai" ;;
    gemma-4-31b)     echo "google/gemma-4-31B-it --tool-call-parser gemma4 $TEXT_ONLY" ;;
    gemma-4-26b-a4b) echo "google/gemma-4-26B-A4B-it --tool-call-parser gemma4 $TEXT_ONLY" ;;
    qwen3.8-27b)     echo "Qwen/Qwen3.8-27B --tool-call-parser qwen3_coder --reasoning-parser qwen3 --gdn-prefill-backend triton $TEXT_ONLY" ;;
    ministral-3-14b) echo "mistralai/Ministral-3-14B-Instruct-2512 --tool-call-parser mistral --tokenizer-mode mistral --config-format mistral --load-format mistral $TEXT_ONLY" ;;
    *) return 1 ;;
  esac
}
run_args() {     # options of replicate.py per model (reasoning effort low where the model has one, as for gpt-oss-20b)
  case "$1" in
    qwen3.8-27b|gpt-oss-20b) echo "--reasoning-effort low" ;;
    *) echo "" ;;
  esac
}

for name in "$@"; do
  log=logs/vllm_$name.log
  echo "== $name: serving ($(date))"
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 5; done
  rm -f "$log"   # a log left by an earlier run would hold an old 'startup complete' line
  vllm serve $(serve_args "$name") --served-model-name "$name" $COMMON > "$log" 2>&1 &
  pid=$!
  until grep -q "Application startup complete" "$log" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  until curl -sf localhost:8011/v1/models > /dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done   # and it answers
  if ! kill -0 $pid 2>/dev/null; then echo "== $name: the server failed, see $log"; tail -n 5 "$log"; continue; fi
  if [ "$PILOT" -gt 0 ]; then
    echo "== $name: pilot"
    python replicate.py --model "$name" --designs "$DESIGNS" --pilot "$PILOT" $(run_args "$name") > logs/replicate_pilot_${name}_$DESIGNS.log 2>&1
    tail -n 7 logs/replicate_pilot_${name}_$DESIGNS.log
  fi
  echo "== $name: designs $DESIGNS"
  python replicate.py --model "$name" --designs "$DESIGNS" $(run_args "$name") > logs/replicate_${name}_$DESIGNS.log 2>&1
  tail -n 20 logs/replicate_${name}_$DESIGNS.log
  kill $pid; wait $pid 2>/dev/null
  echo "== $name: done ($(date))"
done
