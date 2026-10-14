"""Train and compare the lightweight theme classifiers that replace the LLM at serve time.

    uv run --extra translate --extra research python research/theme_classifier.py

Teacher: Qwen3.8-27B's sentence theme labels (data/labels/themes_qwen.jsonl), 3,882 complaint and
praise sentences. Student: sentence embedding -> one-vs-rest logistic regression per theme
(multi-label). Complaint and praise themes are separate problems.

Embedding models compared (multilingual, permissive licences, pinned revisions):
  paraphrase-multilingual-MiniLM-L12-v2 (118M, Apache-2.0)
  multilingual-e5-base                  (278M, MIT; "query: " prefix)
  bge-m3                                (568M, MIT; dense CLS embedding)

Evaluation, built so that it cannot leak:
- outer 5-fold GroupKFold by review: sentences of one review never sit in both train and test;
- per-theme decision thresholds tuned on inner 3-fold out-of-fold predictions of the TRAINING
  part only (grid 0.1..0.9), never on the test fold;
- metrics: micro/macro-F1 over themes; and what matters for the insights, review-level theme
  prevalence vs the teacher (mean absolute error in percentage points) and the Spearman
  correlation of the complaint priority ranking.
The winner is retrained on all data and exported for serving (theme_classifier.py in the package).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data/snapshots/notion_2026-10-13"
OUT = ROOT / "research/results/themes"
EMBEDDERS = {
    "minilm-l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c21122", ""),
    "e5-base": ("intfloat/multilingual-e5-base", "d128750597", "query: "),
    "bge-m3": ("BAAI/bge-m3", "5617a9f61b", ""),
}
THRESHOLDS = np.round(np.arange(0.1, 0.91, 0.05), 2)
C = 4.0


def load() -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
    sentences = [json.loads(line) for line in (SNAP / "sentences.jsonl").read_text().splitlines()]
    labels = {
        json.loads(line)["sid"]: json.loads(line)["themes"]
        for line in (SNAP / "theme_labels.jsonl").read_text().splitlines()
    }
    tax = json.loads((ROOT / "data/themes.json").read_text())
    ids = {k: [t["id"] for t in tax[k]] for k in ("complaint", "praise")}
    rows = [s for s in sentences if s["sid"] in labels and s["kind"] in ("complaint", "praise")]
    return rows, labels, ids


def embed(name: str, texts: list[str]) -> np.ndarray:
    cache = OUT / f"emb_{name}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(texts):
            return arr
    from sentence_transformers import SentenceTransformer

    repo, rev, prefix = EMBEDDERS[name]
    model = SentenceTransformer(repo, revision=rev, device="cpu")
    started = time.time()
    arr = model.encode(
        [prefix + t for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"  {name}: embedded {len(texts)} sentences in {time.time() - started:.0f}s", flush=True)
    np.save(cache, arr)
    return np.asarray(arr)


def fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    probs = np.zeros((x_te.shape[0], y_tr.shape[1]))
    for j in range(y_tr.shape[1]):
        if y_tr[:, j].sum() < 2:
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        probs[:, j] = clf.fit(x_tr, y_tr[:, j]).predict_proba(x_te)[:, 1]
    return probs


def tune_thresholds(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    oof = np.zeros(y.shape)
    for tr, te in GroupKFold(3).split(x, y, groups):
        oof[te] = fit_predict(x[tr], y[tr], x[te])
    best = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        scores = [f1_score(y[:, j], oof[:, j] >= t, zero_division=0) for t in THRESHOLDS]
        best[j] = THRESHOLDS[int(np.argmax(scores))]
    return best


def evaluate(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.zeros(y.shape, dtype=bool)
    for tr, te in GroupKFold(5).split(x, y, groups):
        thr = tune_thresholds(x[tr], y[tr], groups[tr])
        pred[te] = fit_predict(x[tr], y[tr], x[te]) >= thr
    return pred, {
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def review_prevalence(
    rows: list[dict[str, Any]], mat: np.ndarray, theme_ids: list[str], n_reviews: int
) -> dict[str, float]:
    by_review: dict[str, set[str]] = defaultdict(set)
    for r, vec in zip(rows, mat, strict=True):
        by_review[r["review_id"]].update(t for t, on in zip(theme_ids, vec, strict=True) if on)
    return {t: sum(t in s for s in by_review.values()) / n_reviews for t in theme_ids}


def main() -> None:
    rows_all, labels, theme_ids = load()
    n_reviews = len(
        {
            json.loads(line)["review_id"]
            for line in (SNAP / "processed.jsonl").read_text().splitlines()
        }
    )
    texts = [r["text"] for r in rows_all]
    results: dict[str, Any] = {}
    for name in EMBEDDERS:
        emb_all = embed(name, texts)
        res: dict[str, Any] = {}
        for kind in ("complaint", "praise"):
            idx = [i for i, r in enumerate(rows_all) if r["kind"] == kind]
            rows = [rows_all[i] for i in idx]
            x = emb_all[idx]
            y = np.array(
                [[t in labels[r["sid"]] for t in theme_ids[kind]] for r in rows], dtype=int
            )
            groups = np.array([r["review_id"] for r in rows])
            pred, scores = evaluate(x, y, groups)
            teacher = review_prevalence(rows, y.astype(bool), theme_ids[kind], n_reviews)
            student = review_prevalence(rows, pred, theme_ids[kind], n_reviews)
            mae = float(np.mean([abs(teacher[t] - student[t]) for t in theme_ids[kind]]) * 100)
            res[kind] = {**scores, "prevalence_mae_pp": mae, "n_sentences": len(rows)}
            if kind == "complaint":
                rho = spearmanr(
                    [teacher[t] for t in theme_ids[kind]], [student[t] for t in theme_ids[kind]]
                ).statistic
                res[kind]["prevalence_rank_spearman"] = float(rho)
        results[name] = res
        print(
            f"{name:10s} complaint micro-F1 {res['complaint']['micro_f1']:.3f} macro-F1 "
            f"{res['complaint']['macro_f1']:.3f} prevalence MAE "
            f"{res['complaint']['prevalence_mae_pp']:.2f}pp "
            f"rank rho {res['complaint']['prevalence_rank_spearman']:.3f} | praise micro-F1 "
            f"{res['praise']['micro_f1']:.3f} macro-F1 {res['praise']['macro_f1']:.3f}",
            flush=True,
        )
    (OUT / "classifier_comparison.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
