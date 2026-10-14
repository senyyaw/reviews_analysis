#!/bin/bash
# Usage: NEED_MIB=40000 wait_gpu_run.sh <logfile> <command...>
# Waits until the GPU with the MOST free memory has >= NEED_MIB free (other users may be running
# on it too; agreed with the user), then runs the command there.
log=$1; shift
need=${NEED_MIB:-40000}
cd /workspace/reviews && source .venv/bin/activate
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1
while true; do
  read -r G FREE <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr | head -1 | tr , ' ')"
  [ "${FREE:-0}" -ge "$need" ] && break
  sleep 30
done
echo "== $(date +%T) start on GPU $G ($FREE MiB free)" > "$log"
CUDA_VISIBLE_DEVICES=$G "$@" >> "$log" 2>&1
rc=$?  # capture before $(date) runs, which would reset $?
echo "== $(date +%T) exit $rc" >> "$log"
