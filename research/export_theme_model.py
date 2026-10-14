"""Train the final theme classifier on all teacher labels and export it for serving.

    uv run --extra translate --extra research python research/export_theme_model.py

Writes models/themes/:
- embedder.int8.onnx   multilingual-e5-base with mean pooling + L2 normalisation inside the graph,
                       per-channel dynamic int8 (as for the sentiment models)
- tokenizer.json       its tokenizer
- classifier.json      per kind (complaint/praise): theme ids, logistic-regression weights,
                       intercepts and per-theme thresholds (3-fold out-of-fold F1 on all data),
                       plus the taxonomy version and provenance
The classifier is trained on the int8 graph's OWN embeddings, so training and serving see
identical vectors. Training on fp32 embeddings and serving int8 flipped 8% of complaint-sentence
decisions. Cross-validated quality on the int8 embeddings is reported, along with the int8 vs fp32
cosine similarity.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import theme_classifier as tc  # noqa: E402

OUT = ROOT / "models/themes"
REPO, REV, PREFIX = tc.EMBEDDERS["e5-base"]
MAX_LEN = 256


class Pooled(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def export_embedder() -> Tokenizer:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel

    st = SentenceTransformer(REPO, revision=REV, device="cpu")
    model = Pooled(AutoModel.from_pretrained(REPO, revision=REV, dtype=torch.float32).eval()).eval()
    st.tokenizer.save_pretrained(str(OUT / "_tok"))
    (OUT / "_tok" / "tokenizer.json").replace(OUT / "tokenizer.json")
    for f in (OUT / "_tok").iterdir():
        f.unlink()
    (OUT / "_tok").rmdir()
    tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    tok.enable_truncation(MAX_LEN)
    tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
    enc = tok.encode_batch([PREFIX + "example sentence", PREFIX + "another one"])
    ids = torch.tensor([e.ids for e in enc])
    mask = torch.tensor([e.attention_mask for e in enc])
    fp32 = OUT / "embedder.fp32.onnx"
    torch.onnx.export(
        model,
        (ids, mask),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        str(fp32), str(OUT / "embedder.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True
    )
    fp32.unlink()
    return tok


def int8_embed(tok: Tokenizer, texts: list[str]) -> np.ndarray:
    """Embeddings from the shipped int8 graph, one sentence at a time (as served)."""
    sess = ort.InferenceSession(str(OUT / "embedder.int8.onnx"), providers=["CPUExecutionProvider"])
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for i, t in enumerate(texts):
        e = tok.encode(PREFIX + t)
        out[i] = sess.run(
            None, {"input_ids": np.array([e.ids]), "attention_mask": np.array([e.attention_mask])}
        )[0][0]
    return out


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    OUT.mkdir(parents=True, exist_ok=True)
    rows, labels, theme_ids = tc.load()
    texts = [r["text"] for r in rows]
    fp32_emb = tc.embed("e5-base", texts)
    tok = export_embedder()
    q = int8_embed(tok, texts)
    cos = (q * fp32_emb).sum(1)
    n_reviews = len({r["review_id"] for r in rows})

    tax = json.loads((ROOT / "data/themes.json").read_text())
    payload: dict[str, object] = {
        "embedder": REPO,
        "revision": REV,
        "prefix": PREFIX,
        "max_len": MAX_LEN,
        "taxonomy_version": tax["_meta"]["version"],
        "teacher": "Qwen/Qwen3.8-27B-FP8 sentence labels (data/labels/themes_qwen.jsonl)",
        "model": "one-vs-rest logistic regression on the int8 embeddings, C=4, "
        "class_weight=balanced; thresholds from 3-fold out-of-fold F1",
    }
    cv: dict[str, dict[str, float]] = {}
    for kind in ("complaint", "praise"):
        idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
        x = q[idx]
        y = np.array(
            [[t in labels[rows[i]["sid"]] for t in theme_ids[kind]] for i in idx], dtype=int
        )
        groups = np.array([rows[i]["review_id"] for i in idx])
        # cross-validated quality ON THE SERVED EMBEDDINGS (GroupKFold, nested thresholds)
        pred, scores = tc.evaluate(x, y, groups)
        sub = [rows[i] for i in idx]
        teacher = tc.review_prevalence(sub, y.astype(bool), theme_ids[kind], n_reviews)
        student = tc.review_prevalence(sub, pred, theme_ids[kind], n_reviews)
        scores["prevalence_mae_pp"] = float(
            np.mean([abs(teacher[t] - student[t]) for t in theme_ids[kind]]) * 100
        )
        cv[kind] = scores
        thr = tc.tune_thresholds(x, y, groups)
        coef, intercept = [], []
        for j in range(y.shape[1]):
            clf = LogisticRegression(C=tc.C, class_weight="balanced", max_iter=2000).fit(x, y[:, j])
            coef.append(clf.coef_[0].round(6).tolist())
            intercept.append(float(clf.intercept_[0]))
        payload[kind] = {
            "themes": theme_ids[kind],
            "coef": coef,
            "intercept": intercept,
            "thresholds": thr.tolist(),
            "n_train": len(idx),
        }
    payload["cross_validation_on_int8"] = cv
    payload["export_check"] = {
        "cosine_int8_vs_fp32_min": float(cos.min()),
        "cosine_int8_vs_fp32_mean": float(cos.mean()),
        "size_mb": round((OUT / "embedder.int8.onnx").stat().st_size / 1e6, 1),
    }
    (OUT / "classifier.json").write_text(json.dumps(payload))
    print(json.dumps({"cv": cv, "export": payload["export_check"]}, indent=1))


if __name__ == "__main__":
    main()
