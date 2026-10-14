"""TopicGPT stage 2: consolidate extracted aspects into a theme taxonomy, as a DRAFT for a human.

    python llm_consolidate.py --aspects aspect_counts.json --output out/taxonomy_draft_gemma.json

Input: {"complaint": [[aspect, count], ...], "praise": [...]}, the extracted aspect labels with
their frequency. The LLM proposes 8-14 themes per side for a product team. Each theme gets an id,
a name, an inclusion/exclusion definition and the aspects it covers. A human then approves,
merges or renames the themes, which become themes.json.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

PROMPT = """You are helping the product team of Notion (the note-taking / workspace app) understand
App Store reviews. Below are {kind} aspects extracted from review sentences, with how many
sentences mention each (format "count  aspect").

Group them into {lo}-{hi} THEMES that a product team can act on. Requirements:
- Themes must be distinct and non-overlapping; merge near-duplicates ("app stability",
  "app freezing", "app crashes" -> one theme), but keep genuinely different problems apart
  (e.g. iPad-specific problems vs AI features vs pricing).
- Prefer themes that map to an owner/action (a feature area, platform, or policy).
- Each theme: id (snake_case, prefixed "{prefix}_"), name (2-5 words), definition (1-2 sentences:
  what belongs in it and what does NOT), and up to 10 of its most frequent example aspects
  (verbatim from the list).
- Cover the frequent aspects; rare idiosyncratic ones may stay unassigned.

Aspects:
{listing}

Answer with JSON only."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspects", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--revision", default="017b9c7af6")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    aspects = json.loads(Path(args.aspects).read_text())
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["themes"],
        "properties": {
            "themes": {
                "type": "array",
                "minItems": 6,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "name", "definition", "aspects"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "definition": {"type": "string"},
                        "aspects": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
                    },
                },
            }
        },
    }
    llm = LLM(
        model=args.model,
        revision=args.revision,
        max_model_len=16384,  # Qwen: ~64 KiB KV per token, so 16k context needs ~1 GiB
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        max_num_seqs=4,  # one prompt at a time; Qwen's hybrid layers need a cache block per seq
        language_model_only=True,
        generation_config="vllm",
    )
    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=6000,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    result = {}
    for kind, prefix, lo, hi in (("complaint", "c", 10, 14), ("praise", "p", 6, 10)):
        listing = "\n".join(f"{n:4d}  {a}" for a, n in aspects[kind])
        msg = PROMPT.format(kind=kind, lo=lo, hi=hi, prefix=prefix, listing=listing)
        out = llm.chat(
            [[{"role": "user", "content": msg}]],
            params,
            chat_template_kwargs={"enable_thinking": False},
        )
        text = out[0].outputs[0].text
        Path(args.output).with_suffix(f".{kind}.raw.txt").write_text(text)  # kept for audit
        result[kind] = json.loads(text)["themes"]
    Path(args.output).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("ok", {k: len(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
