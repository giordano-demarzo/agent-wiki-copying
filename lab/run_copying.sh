#!/usr/bin/env bash
# Fig. 2 (copying) on local models, one after the other on one GPU: serve the model with vLLM, run the 820 stimuli
# of data/trials.jsonl with run.py (temperature 1, no reasoning-effort setting, as for Gemma 4 31B and Qwen3.8 27B),
# stop the server. Replies in data/replies_<model>.jsonl, logs in logs/.
#     bash run_copying.sh gemma-4-26b-a4b ministral-3-14b
set -u
cd "$(dirname "$0")"
mkdir -p logs
source /opt/conda/etc/profile.d/conda.sh && conda activate llm-vllm
TEXT_ONLY="--limit-mm-per-prompt {\"image\":0,\"video\":0}"
serve_args() {
  case "$1" in
    gemma-4-26b-a4b) echo "google/gemma-4-26B-A4B-it $TEXT_ONLY" ;;
    gemma-4-31b)     echo "google/gemma-4-31B-it $TEXT_ONLY" ;;
    ministral-3-14b) echo "mistralai/Ministral-3-14B-Instruct-2512 --tokenizer-mode mistral --config-format mistral --load-format mistral $TEXT_ONLY" ;;
    *) return 1 ;;
  esac
}
for name in "$@"; do
  log=logs/vllm_copying_$name.log
  rm -f "$log"
  echo "== $name: serving ($(date))"
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 5; done
  vllm serve $(serve_args "$name") --served-model-name "$name" --port 8011 --max-model-len 32768 \
      --gpu-memory-utilization 0.92 > "$log" 2>&1 &
  pid=$!
  until curl -sf localhost:8011/v1/models > /dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  if ! kill -0 $pid 2>/dev/null; then echo "== $name: the server failed, see $log"; continue; fi
  echo "== $name: trials"
  python run.py --model "$name" --base-url http://localhost:8011/v1 --out "data/replies_$name.jsonl" \
      --concurrency 64 --temperature 1.0 --reasoning-effort= > "logs/copying_$name.log" 2>&1
  python -c "
import json; r=[json.loads(l) for l in open('data/replies_$name.jsonl')]
print('== $name:', len(r), 'replies,', sum(x['reply'] is None for x in r), 'unparsed')"
  kill $pid; wait $pid 2>/dev/null
done
