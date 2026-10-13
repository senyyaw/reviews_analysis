#!/bin/bash
# Label the other apps' reviews (distillation training data) with both annotators on the freest GPU.
set -uo pipefail
cd /workspace/reviews && source .venv/bin/activate
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1
pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | awk -F', ' '$2 < 500 {print $1; exit}'; }
run() {
  gpu=$(pick_gpu); [ -z "$gpu" ] && { echo "NO FREE GPU"; return 1; }
  echo "== $(date +%T) $1 on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python llm_label.py --model "$2" --revision "$3" --max-num-seqs 64 \
     --input others_llm_input.jsonl --output out/others_labels_$1.jsonl --gpu-memory-utilization 0.90 > out/others_$1.log 2>&1
  rc=$?; echo "== $(date +%T) $1 exit $rc"; tail -1 out/others_$1.log; return $rc
}
{ run qwen3.8-27b Qwen/Qwen3.8-27B-FP8 017b9c7af6; run gemma4-31b RedHatAI/gemma-4-31B-it-FP8-dynamic d4ab4f579d; echo "== OTHERS DONE"; } > out/others_run.log 2>&1
