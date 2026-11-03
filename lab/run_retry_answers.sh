#!/usr/bin/env bash
# A further retry of the answer experiments (after run_fill_gaps.sh). Fill the gaps of the laboratory runs, one model at a time on the GPU: trials of the copying experiment whose reply
# did not parse, and agents of the answer experiments with an error or without a readable answer, run again; the
# failed records move to *_failed.jsonl, so every trial keeps one record. Ministral 3 14B writes broken JSON in
# about half of its free-format copying replies, so its copying run is redone in full with the output constrained
# to a JSON object (the free-format run is kept as data/replies_ministral-3-14b_freeformat.jsonl).
#     bash run_retry_answers.sh
set -u
cd "$(dirname "$0")"
mkdir -p logs
source /opt/conda/etc/profile.d/conda.sh && conda activate llm-vllm
TEXT_ONLY="--limit-mm-per-prompt {\"image\":0,\"video\":0}"
serve() {   # name, model and options
  local name=$1; shift
  local log=logs/vllm_fill_$name.log
  rm -f "$log"
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 5; done
  vllm serve "$@" --served-model-name "$name" --port 8011 --max-model-len 32768 --gpu-memory-utilization 0.92 \
      --enable-auto-tool-choice > "$log" 2>&1 &
  PID=$!
  until curl -sf localhost:8011/v1/models > /dev/null || ! kill -0 $PID 2>/dev/null; do sleep 5; done
  kill -0 $PID 2>/dev/null || { echo "== $name: the server failed, see $log"; return 1; }
}
stop() { kill $PID; wait $PID 2>/dev/null; }
URL=http://localhost:8011/v1

for spec in "ministral-3-14b|mistralai/Ministral-3-14B-Instruct-2512 --tool-call-parser mistral --tokenizer-mode mistral --config-format mistral --load-format mistral $TEXT_ONLY|" \
            "qwen3.8-27b|Qwen/Qwen3.8-27B --tool-call-parser qwen3_coder --reasoning-parser qwen3 --gdn-prefill-backend triton $TEXT_ONLY|--reasoning-effort low" \
            "gpt-oss-20b|openai/gpt-oss-20b --tool-call-parser openai|--reasoning-effort low"; do
  IFS='|' read -r name args rargs <<< "$spec"
  echo "== $name ($(date))"
  if serve $name $args; then
    python replicate.py --model $name --designs ABPR --retry $rargs > logs/retry2_answers_$name.log 2>&1
    tail -n 1 logs/retry2_answers_$name.log | head -1
    grep "set aside\|agents, " logs/retry2_answers_$name.log
    stop
  fi
done
