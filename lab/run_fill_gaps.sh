#!/usr/bin/env bash
# Fill the gaps of the laboratory runs, one model at a time on the GPU: trials of the copying experiment whose reply
# did not parse, and agents of the answer experiments with an error or without a readable answer, run again; the
# failed records move to *_failed.jsonl, so every trial keeps one record. Ministral 3 14B writes broken JSON in
# about half of its free-format copying replies, so its copying run is redone in full with the output constrained
# to a JSON object (the free-format run is kept as data/replies_ministral-3-14b_freeformat.jsonl).
#     bash run_fill_gaps.sh
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

echo "== ministral-3-14b ($(date))"
if serve ministral-3-14b mistralai/Ministral-3-14B-Instruct-2512 --tool-call-parser mistral --tokenizer-mode mistral \
        --config-format mistral --load-format mistral $TEXT_ONLY; then
  [ -f data/replies_ministral-3-14b_freeformat.jsonl ] || mv data/replies_ministral-3-14b.jsonl data/replies_ministral-3-14b_freeformat.jsonl
  python run.py --model ministral-3-14b --base-url $URL --out data/replies_ministral-3-14b.jsonl --concurrency 64 \
      --temperature 1.0 --reasoning-effort= --json-mode > logs/fill_copying_ministral.log 2>&1
  python run.py --model ministral-3-14b --base-url $URL --out data/replies_ministral-3-14b.jsonl --concurrency 64 \
      --temperature 1.0 --reasoning-effort= --json-mode --retry >> logs/fill_copying_ministral.log 2>&1
  python replicate.py --model ministral-3-14b --designs ABPR --retry > logs/fill_answers_ministral.log 2>&1
  stop
fi
echo "== qwen3.8-27b ($(date))"
if serve qwen3.8-27b Qwen/Qwen3.8-27B --tool-call-parser qwen3_coder --reasoning-parser qwen3 --gdn-prefill-backend triton \
        $TEXT_ONLY; then
  python replicate.py --model qwen3.8-27b --designs ABPR --retry --reasoning-effort low > logs/fill_answers_qwen.log 2>&1
  stop
fi
echo "== gpt-oss-20b ($(date))"
if serve gpt-oss-20b openai/gpt-oss-20b --tool-call-parser openai; then
  python run.py --model gpt-oss-20b --base-url $URL --out data/replies_gpt-oss-20b.jsonl --concurrency 64 \
      --temperature 1.0 --retry > logs/fill_copying_gpt-oss.log 2>&1
  stop
fi
echo "== done ($(date))"
for f in data/replies_*.jsonl; do
  case $f in *_failed.jsonl|*_freeformat.jsonl) continue ;; esac
  python -c "import json,sys; r=[json.loads(l) for l in open('$f')]; print('$f', len(r), 'replies,', sum(x['reply'] is None for x in r), 'unparsed')"
done
