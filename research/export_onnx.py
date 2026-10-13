"""Export the chosen fine-tuned models to int8 ONNX and verify the export changes nothing important.

    uv run --extra translate --extra research python research/export_onnx.py \
        --checkpoints ~/Downloads/models

For each model:
1. rebuild the multi-task network (encoder + linear(4) head, as in finetune.py) and load best.pt;
2. export to ONNX (fp32), then apply dynamic int8 quantisation of the weights (onnxruntime);
3. compare PyTorch fp32 with ONNX int8 on every Notion review in the pool (1,143): sentiment label
   agreement, complaint agreement, and the maximum absolute difference in probabilities;
4. write models/<name>/ with model.int8.onnx, the tokenizer files and model.json (labels, source,
   training provenance and agreement figures).
The API serves only the int8 files: onnxruntime + tokenizers, with no torch at runtime.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from finetune import MultiTask  # noqa: E402

MODELS = {
    "sentiment-en": {
        "checkpoint": "cardiff_roberta",
        "base": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "revision": "3216a57f2a",
        "run": "cardiff-roberta-base_s1",
        "languages": ["en", "und"],
    },
    "sentiment-multi": {
        "checkpoint": "clapai",
        "base": "clapAI/roberta-base-multilingual-sentiment",
        "revision": "5986405bd6",
        "run": "clapai-xlmr-base_s0",
        "languages": ["*"],
    },
}
MAX_LEN = 320


class Wrapper(torch.nn.Module):
    def __init__(self, inner: MultiTask) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids=input_ids, attention_mask=attention_mask)


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    sent = np.exp(logits[:, :3] - logits[:, :3].max(1, keepdims=True))
    sent /= sent.sum(1, keepdims=True)
    comp = 1 / (1 + np.exp(-logits[:, 3:]))
    return np.concatenate([sent, comp], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    args = ap.parse_args()
    ckpt_root = Path(args.checkpoints).expanduser()
    pool = [
        json.loads(line)
        for line in (ROOT / "data/snapshots/notion_2026-10-13/processed.jsonl")
        .read_text()
        .splitlines()
    ]
    texts = [p["text_model"] for p in pool]

    for name, cfg in MODELS.items():
        src = ckpt_root / cfg["checkpoint"]
        out = ROOT / "models" / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {name}: loading {src}/best.pt", flush=True)
        model = MultiTask(cfg["base"], cfg["revision"])
        model.load_state_dict(torch.load(src / "best.pt", map_location="cpu", weights_only=True))
        model.eval()
        wrapped = Wrapper(model).eval()

        tok = Tokenizer.from_file(str(src / "tokenizer.json"))
        tok.enable_truncation(MAX_LEN)
        pad_id = tok.token_to_id("<pad>")
        tok.enable_padding(pad_id=pad_id, pad_token="<pad>")

        def encode(batch: list[str], tok: Tokenizer = tok) -> tuple[np.ndarray, np.ndarray]:
            enc = tok.encode_batch(batch)
            return (
                np.array([e.ids for e in enc], dtype=np.int64),
                np.array([e.attention_mask for e in enc], dtype=np.int64),
            )

        ids, mask = encode(texts[:4])
        fp32_path = out / "model.fp32.onnx"
        torch.onnx.export(
            wrapped,
            (torch.from_numpy(ids), torch.from_numpy(mask)),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        int8_path = out / "model.int8.onnx"
        quantize_dynamic(
            str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, per_channel=True
        )

        sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        torch_p, onnx_p = [], []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                ids, mask = encode(texts[i : i + 32])
                torch_p.append(
                    probs_from_logits(
                        wrapped(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
                    )
                )
                onnx_p.append(
                    probs_from_logits(sess.run(None, {"input_ids": ids, "attention_mask": mask})[0])
                )
        tp, op = np.concatenate(torch_p), np.concatenate(onnx_p)
        agreement = {
            "n_reviews": len(texts),
            "sentiment_label_agreement": float((tp[:, :3].argmax(1) == op[:, :3].argmax(1)).mean()),
            "complaint_agreement": float(((tp[:, 3] >= 0.5) == (op[:, 3] >= 0.5)).mean()),
            "max_abs_prob_diff": float(np.abs(tp - op).max()),
            "mean_abs_prob_diff": float(np.abs(tp - op).mean()),
        }
        print(json.dumps(agreement), flush=True)

        for f in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy(src / f, out / f)
        meta = json.loads((ROOT / "research/results/ft" / cfg["run"] / "meta.json").read_text())
        (out / "model.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "outputs": {
                        "logits[0:3]": ["negative", "neutral", "positive"],
                        "logits[3]": "complaint (sigmoid)",
                    },
                    "max_len": MAX_LEN,
                    "pad_token": "<pad>",
                    "languages": cfg["languages"],
                    "base_model": cfg["base"],
                    "base_revision": cfg["revision"],
                    "fine_tuning": {
                        k: meta[k]
                        for k in (
                            "seed",
                            "lr",
                            "epochs",
                            "batch_size",
                            "train_n",
                            "val_n",
                            "best_epoch",
                            "best_val_macro_f1",
                        )
                    },
                    "training_data": "LLM-consensus labels (Qwen3.8-27B + Gemma-4-31B) on 14 other "
                    "productivity apps' App Store reviews; Notion never used",
                    "quantisation": "onnxruntime dynamic int8, per-channel weights",
                    "fp32_vs_int8": agreement,
                    "size_mb": round(int8_path.stat().st_size / 1e6, 1),
                },
                indent=1,
            )
        )
        fp32_path.unlink()  # only the int8 file is shipped
        print(f"   wrote {out} ({int8_path.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
