#!/bin/bash
# Run every (model, seed) fine-tuning job, one per FREE GPU at a time, on a shared node.
# A GPU counts as free if it is idle (<10% utilisation), has <4 GiB in use (other users' idle
# notebooks often hold a little memory), and we have not already launched a job on it.
set -uo pipefail
cd /workspace/reviews && source .venv/bin/activate
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEEDS="0 1 2"
declare -A busy   # gpu -> pid
free_gpu() {
  for line in $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '); do
    IFS=, read -r g m u <<< "$line"
    if [ "$m" -lt 4096 ] && [ "$u" -lt 10 ] && [ -z "${busy[$g]:-}" ]; then echo $g; return; fi
  done
}
reap() { for g in "${!busy[@]}"; do kill -0 ${busy[$g]} 2>/dev/null || unset "busy[$g]"; done; }
{
echo "== $(date +%T) start"
while IFS=$'\t' read -r repo rev lr name; do
  for seed in $SEEDS; do
    out=out/ft/${name}_s$seed
    [ -f $out/meta.json ] && { echo "skip $name s$seed (done)"; continue; }
    while true; do reap; g=$(free_gpu); [ -n "$g" ] && break; sleep 20; done
    mkdir -p $out
    CUDA_VISIBLE_DEVICES=$g nohup python finetune.py --model $repo --revision $rev --lr $lr \
        --seed $seed --out $out > $out/train.log 2>&1 &
    busy[$g]=$!
    echo "== $(date +%T) $name s$seed -> GPU $g (pid $!)"
    sleep 45   # let the job allocate memory before the next free-GPU check
  done
done < ft_jobs.tsv
wait
echo "== $(date +%T) ALL FINETUNE DONE"
} >> out/finetune_run.log 2>&1
