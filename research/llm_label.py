"""LLM annotator: label reviews with a local open-weight model (vLLM, offline batch, GPU).

    python llm_label.py --model Qwen/Qwen3.8-27B-FP8 --revision <sha> \
        --input D1_llm_input.jsonl --output llm_labels_qwen.jsonl

The instructions are the approved human labelling guide (research/labelling_guide.md) minus the
tool-specific sections, so humans and the LLM label against identical definitions. Settings follow
the annotator literature (Törnberg 2024; vLLM reproducibility docs): temperature 0, fixed seed,
thinking disabled, JSON schema enforced by structured decoding, model revision pinned and logged.
The probability of each sentiment is read from the logprobs at the sentiment value's first token,
and serves as the annotator's confidence.

Runs on the GPU server only. It needs vllm, and is not part of the package's dependencies.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # deterministic offline runs
# Greedy decoding needs no top-k/top-p kernel. FlashInfer's sampler JIT-compiles with nvcc,
# which the container lacks, so use vLLM's PyTorch sampler instead.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "complaint": {"type": "boolean"},
    },
    "required": ["sentiment", "complaint"],
}
OUTPUT_INSTRUCTIONS = """
## Your task

You are an annotator following the guide above. You receive one review (title and text, and an
English translation if it is not in English). Star ratings are not given.

Answer with JSON only, in exactly this form:
{"sentiment": "positive" | "neutral" | "negative", "complaint": true | false}
"""
DROP_SECTIONS = ("## When unsure", "## In the labelling tool")


def build_instructions(guide_md: str) -> str:
    """The guide without its tool-specific sections, plus the output format."""
    text = guide_md.split("\n", 1)[1]  # drop the H1 title
    text = re.sub(r"^\*\*Status:.*?\n\n", "", text, flags=re.S | re.M)
    for header in DROP_SECTIONS:
        start = text.find(header)
        if start != -1:
            nxt = text.find("\n## ", start + 1)
            text = text[:start] + (text[nxt + 1 :] if nxt != -1 else "")
    return text.strip() + "\n" + OUTPUT_INSTRUCTIONS


def user_message(item: dict[str, Any]) -> str:
    msg = f"Review:\n{item['text']}"
    if item.get("translation"):
        msg += f"\n\nEnglish translation ({item['lang']}):\n{item['translation']}"
    return msg


def sentiment_probs(output: Any) -> dict[str, float] | None:
    """Probabilities of the three labels at the first token of the sentiment value."""
    tokens = output.outputs[0].logprobs or []
    text = ""
    for step in tokens:
        if re.search(r'"sentiment"\s*:\s*"$', text):
            mass: dict[str, float] = {}
            for lp in step.values():
                tok = (lp.decoded_token or "").strip().lower()
                for label in ("positive", "neutral", "negative"):
                    if tok and label.startswith(tok):
                        mass[label] = mass.get(label, 0.0) + math.exp(lp.logprob)
            total = sum(mass.values())
            return {k: v / total for k, v in mass.items()} if total else None
        chosen = max(step.values(), key=lambda lp: lp.logprob)
        text += chosen.decoded_token or ""
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--guide", default="labelling_guide.md")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)  # longest D1 prompt ~2.5k tokens
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    items = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    instructions = build_instructions(Path(args.guide).read_text(encoding="utf-8"))
    conversations = [
        [{"role": "system", "content": instructions}, {"role": "user", "content": user_message(it)}]
        for it in items
    ]
    llm = LLM(
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        seed=0,
        language_model_only=True,  # text only: skip loading the vision/audio towers
        generation_config="vllm",  # ignore the model's sampling defaults (e.g. Qwen top_k=20)
    )
    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=40,
        logprobs=10,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
    )
    started = time.time()
    outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
    elapsed = time.time() - started

    failures = 0
    with Path(args.output).open("w", encoding="utf-8") as fh:
        for item, out in zip(items, outputs, strict=True):
            raw = out.outputs[0].text
            try:
                label = json.loads(raw)
            except json.JSONDecodeError:
                failures += 1
                label = {"sentiment": None, "complaint": None}
            fh.write(
                json.dumps(
                    {
                        "review_id": item["review_id"],
                        "sentiment": label.get("sentiment"),
                        "complaint": label.get("complaint"),
                        "sentiment_probs": sentiment_probs(out),
                        "raw": raw,
                        "annotator": args.model,
                    }
                )
                + "\n"
            )
    meta = {
        "model": args.model,
        "revision": args.revision,
        "vllm": vllm.__version__,
        "n": len(items),
        "json_failures": failures,
        "seconds": round(elapsed, 1),
        "temperature": 0.0,
        "seed": 0,
        "thinking": False,
        "instructions_chars": len(instructions),
    }
    Path(args.output).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
