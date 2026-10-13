"""Fine-tune a transformer for sentiment (3 classes) + complaint (yes/no) by distilling LLM labels.

    python finetune.py --model microsoft/deberta-v3-base --revision 8ccc9b6f36 --seed 0 \
        --out out/ft/x

Runs on the GPU server.
- Training data: reviews of 14 other productivity apps (never Notion), excluding the D2 dev set,
  keeping only reviews where both LLM annotators agree on sentiment AND complaint.
- A 10% stratified split of that data is the validation set: model selection uses the epoch with
  the best validation sentiment macro-F1, plus seeds.
- The script never sees Notion gold labels. It writes probabilities for D1 and D2, and the
  evaluation runs elsewhere with the same code as the off-the-shelf comparison.

Architecture: pretrained encoder -> first-token representation -> dropout -> linear(4).
Logits 0-2 are sentiment (cross-entropy with inverse-sqrt class weights, because neutral is rare).
Logit 3 is complaint (binary cross-entropy). The loss is their sum.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

CLASSES = ["negative", "neutral", "positive"]


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        (row := json.loads(line))["review_id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def user_text(item: dict[str, Any]) -> str:
    return item["text"]  # original language; the encoder sees exactly what it sees at inference


class MultiTask(nn.Module):
    def __init__(self, name: str, revision: str) -> None:
        super().__init__()
        # transformers 5 loads checkpoints in their saved dtype; DeBERTa-v3 is stored in fp16, which
        # overflows to NaN when trained without loss scaling. Keep master weights in fp32 and use
        # bf16 autocast for compute only.
        self.encoder = AutoModel.from_pretrained(name, revision=revision, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 4)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        out = self.encoder(**batch).last_hidden_state[:, 0]
        return self.head(self.dropout(out))


def batches(n: int, size: int, shuffle: bool, rng: random.Random) -> list[list[int]]:
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    return [idx[i : i + size] for i in range(0, n, size)]


@torch.no_grad()
def predict(model: MultiTask, tok: Any, texts: list[str], max_len: int, device: str) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(
            texts[i : i + 64],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).float()
        sent = torch.softmax(logits[:, :3], -1)
        comp = torch.sigmoid(logits[:, 3:])
        out.append(torch.cat([sent, comp], 1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = Path(args.data_dir)

    # --- data: LLM-consensus labels on other apps, minus D2 ---
    texts = read_jsonl(data / "others_llm_input.jsonl")
    qwen = read_jsonl(data / "out/others_labels_qwen3.8-27b.jsonl")
    gemma = read_jsonl(data / "out/others_labels_gemma4-31b.jsonl")
    d2 = set(json.loads((data / "D2_dev_set.json").read_text())["review_ids"])
    examples = [
        (texts[r]["text"], CLASSES.index(qwen[r]["sentiment"]), float(qwen[r]["complaint"]))
        for r in sorted(texts)
        if r not in d2
        and qwen[r]["sentiment"]
        and gemma[r]["sentiment"]
        and qwen[r]["sentiment"] == gemma[r]["sentiment"]
        and qwen[r]["complaint"] == gemma[r]["complaint"]
    ]
    # Stratified 90/10 split with a FIXED seed (the same split for every model and seed).
    split_rng = random.Random(1234)
    by_class: dict[int, list[int]] = {}
    for i, (_, y, _) in enumerate(examples):
        by_class.setdefault(y, []).append(i)
    val_idx: set[int] = set()
    for idxs in by_class.values():
        val_idx |= set(split_rng.sample(idxs, max(1, len(idxs) // 10)))
    train = [examples[i] for i in range(len(examples)) if i not in val_idx]
    val = [examples[i] for i in sorted(val_idx)]
    counts = np.bincount([y for _, y, _ in train], minlength=3)
    class_w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float)
    class_w = class_w / class_w.mean()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = MultiTask(args.model, args.revision).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    ce = nn.CrossEntropyLoss(weight=class_w.to(device))
    bce = nn.BCEWithLogitsLoss()

    eval_sets = {
        name: read_jsonl(data / fname)
        for name, fname in (("D1", "D1_llm_input.jsonl"), ("D2", "others_llm_input.jsonl"))
    }
    eval_sets["D2"] = {r: eval_sets["D2"][r] for r in sorted(d2)}

    best_f1, best_epoch, log = -1.0, -1, []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        for b in batches(len(train), args.batch_size, True, rng):
            enc = tok(
                [train[i][0] for i in b],
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt",
            ).to(device)
            y = torch.tensor([train[i][1] for i in b], device=device)
            c = torch.tensor([train[i][2] for i in b], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).float()
            loss = ce(logits[:, :3], y) + bce(logits[:, 3], c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
        p = predict(model, tok, [t for t, _, _ in val], args.max_len, device)
        f1 = f1_score([y for _, y, _ in val], p[:, :3].argmax(1), average="macro")
        comp_f1 = f1_score([c for _, _, c in val], (p[:, 3] >= 0.5).astype(float))
        log.append(
            {
                "epoch": epoch + 1,
                "val_sentiment_macro_f1": f1,
                "val_complaint_f1": comp_f1,
                "loss_last": float(loss),
            }
        )
        print(json.dumps(log[-1]), flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            preds = {"val": p}
            for name, items in eval_sets.items():
                ids = sorted(items)
                probs = predict(
                    model, tok, [user_text(items[r]) for r in ids], args.max_len, device
                )
                preds[name] = dict(zip(ids, probs.tolist(), strict=True))
            torch.save(model.state_dict(), out_dir / "best.pt")

    with (out_dir / "preds.json").open("w") as fh:
        json.dump(
            {
                "D1": preds["D1"],
                "D2": preds["D2"],
                "val": preds["val"].tolist(),
                "val_gold": [[y, c] for _, y, c in val],
            },
            fh,
        )
    meta = {
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "train_n": len(train),
        "val_n": len(val),
        "train_class_counts": counts.tolist(),
        "class_weights": class_w.tolist(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "log": log,
        "seconds": round(time.time() - started, 1),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    tok.save_pretrained(out_dir / "tokenizer")


if __name__ == "__main__":
    main()
