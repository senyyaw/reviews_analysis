#!/bin/bash
set -euo pipefail
cd /workspace/reviews
export HF_HOME=/workspace/hf_cache
{
echo "== $(date) installing gcc (Triton/Inductor compile kernels at runtime)"
DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++
echo "== $(date) installing uv + vllm"
python -m pip install --user -q uv
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --python 3.11 /workspace/reviews/.venv
source /workspace/reviews/.venv/bin/activate
uv pip install -q "vllm==0.31.0" "huggingface_hub[hf_transfer]" --torch-backend=auto
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda, 'cuda ok', torch.cuda.is_available())"
echo "== $(date) downloading models"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6 > /dev/null
hf download RedHatAI/gemma-4-31B-it-FP8-dynamic --revision d4ab4f579d > /dev/null
du -sh $HF_HOME
echo "== $(date) SETUP DONE"
} > /workspace/reviews/setup.log 2>&1
