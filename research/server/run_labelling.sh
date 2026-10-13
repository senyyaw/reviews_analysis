#!/bin/bash
# Label D1 with both LLM annotators on the freest GPU (shared node: never take a busy one).
set -uo pipefail
command -v gcc >/dev/null || { echo "gcc missing: apt-get install -y gcc g++"; exit 1; }
cd /workspace/reviews && source .venv/bin/activate
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | awk -F', ' '$2 < 500 {print $1; exit}'; }
run() {  # name model revision
  gpu=$(pick_gpu); if [ -z "$gpu" ]; then echo "NO FREE GPU"; return 1; fi
  echo "== $(date +%T) $1 on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python llm_label.py --model "$2" --revision "$3" \
     --input D1_llm_input.jsonl --output out/llm_labels_$1.jsonl --gpu-memory-utilization 0.90 > out/$1.log 2>&1
  rc=$?  # capture before anything else runs
  echo "== $(date +%T) $1 exit $rc"; tail -1 out/$1.log; return $rc
}
{
run qwen3.8-27b Qwen/Qwen3.8-27B-FP8 017b9c7af6
run gemma4-31b RedHatAI/gemma-4-31B-it-FP8-dynamic d4ab4f579d
echo "== LABELLING DONE"
} > out/run.log 2>&1
