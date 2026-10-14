"""TopicGPT-style topic modelling with a local LLM (Pham et al., NAACL 2024), on the GPU server.

    python llm_topics.py extract  --input sentences_for_llm.jsonl --output out/aspects_qwen.jsonl
    python llm_topics.py assign   --input sentences_for_llm.jsonl --taxonomy themes.json \
                                  --model ... --output out/themes_qwen.jsonl

Stage 1 "extract": for every complaint or praise sentence (given with its full review for
context), name the specific product aspect in 2-5 words, or "none" when the sentence is generic
("This is very frustrating."). Embedding clustering grouped sentences by tone rather than aspect
(silhouette 0.02-0.04), which is why the LLM does this step.
Stage 2, consolidating aspects into a taxonomy, happens outside this script: aspect counts go to
the LLM once, and a human approves the result.
Stage 3 "assign": tag each sentence with zero or more themes from the approved taxonomy (the
enum is enforced by structured decoding). Two model families are run so their agreement can be
measured.

Settings as in llm_label.py: temperature 0, thinking off, pinned revisions, JSON schema enforced.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

EXTRACT_SYSTEM = """You analyse App Store reviews of Notion, the note-taking and workspace app.
You get one review and ONE sentence from it. Name the specific product aspect that this sentence
{verb}, as a short noun phrase of 2-5 words, in English, lowercase.

Rules:
- Be specific and concrete: "ipad typing lag", "ai features forced", "offline mode missing",
  "price of plus plan", "templates", "page loading speed", "sync between devices".
- Use the review only as context; the aspect must be what THIS sentence is about.
- If the sentence names no concrete aspect (pure emotion, a rating, a general statement such as
  "This is very frustrating." or "Love it!"), answer "none".
- Do not invent details that are not in the text.

Answer with JSON only: {{"aspect": "<2-5 words or none>"}}"""

ASSIGN_SYSTEM = """You analyse App Store reviews of Notion, the note-taking and workspace app.
You get one review and ONE sentence from it, plus a list of themes with definitions. Choose
EVERY theme that the sentence is about (usually one, sometimes two), or none if no theme fits.
Judge the sentence; use the review only as context.

Themes:
{themes}

Answer with JSON only: {{"themes": [<theme ids>]}}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def user_message(item: dict[str, Any]) -> str:
    return f"Review:\n{item['review']}\n\nSentence:\n{item['text']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "assign"])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--revision", default="017b9c7af6")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--taxonomy", help="themes.json (stage assign)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    items = read_jsonl(Path(args.input))
    if args.stage == "extract":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["aspect"],
            "properties": {"aspect": {"type": "string", "maxLength": 60}},
        }
        systems = {
            "complaint": EXTRACT_SYSTEM.format(verb="complains about or criticises"),
            "praise": EXTRACT_SYSTEM.format(verb="praises or likes"),
        }
        convs = [
            [
                {"role": "system", "content": systems[it["kind"]]},
                {"role": "user", "content": user_message(it)},
            ]
            for it in items
        ]
        max_tokens = 40
    else:
        taxonomy = json.loads(Path(args.taxonomy).read_text())
        convs = []
        for it in items:
            themes = taxonomy[it["kind"]]
            listing = "\n".join(f"- {t['id']}: {t['name']}. {t['definition']}" for t in themes)
            convs.append(
                [
                    {"role": "system", "content": ASSIGN_SYSTEM.format(themes=listing)},
                    {"role": "user", "content": user_message(it)},
                ]
            )
        ids = sorted({t["id"] for side in taxonomy.values() for t in side})
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["themes"],
            "properties": {
                "themes": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": ids}}
            },
        }
        max_tokens = 60

    llm = LLM(
        model=args.model,
        revision=args.revision,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_num_seqs=64,
        seed=0,
        language_model_only=True,
        generation_config="vllm",
    )
    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    started = time.time()
    outputs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    failures = 0
    with Path(args.output).open("w", encoding="utf-8") as fh:
        for it, out in zip(items, outputs, strict=True):
            try:
                value = json.loads(out.outputs[0].text)
            except json.JSONDecodeError:
                failures += 1
                value = {}
            fh.write(
                json.dumps({"sid": it["sid"], **value, "model": args.model}, ensure_ascii=False)
                + "\n"
            )
    meta = {
        "stage": args.stage,
        "model": args.model,
        "revision": args.revision,
        "vllm": vllm.__version__,
        "n": len(items),
        "json_failures": failures,
        "seconds": round(time.time() - started, 1),
    }
    Path(args.output).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
