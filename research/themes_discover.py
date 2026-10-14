"""Theme discovery: cluster complaint and praise SENTENCES, then describe each cluster.

    uv run --extra translate --extra research python research/themes_discover.py

1. Sentences: every review in the pool (exact duplicates excluded) is split into sentences in its
   original language. Sentences of fewer than 3 words are skipped.
2. Each sentence gets sentiment and complaint scores from the fine-tuned router (sentiment.py).
   - complaint sentence: label negative, or P(complaint) >= 0.5;
   - praise sentence: label positive and P(complaint) < 0.5.
   A review is often mixed ("Love it. But the iPad app crashes"), so a review-level label would
   put praise into complaint clusters.
3. Embeddings: multilingual-e5-base, used for discovery only. The final embedding model is chosen
   later on labelled sentences. Vectors are L2-normalised.
4. Clustering: Ward agglomerative clustering on the normalised vectors (Euclidean on the unit
   sphere is monotone in cosine). k is chosen by cosine silhouette over K_RANGE. Ward is used rather
   than average linkage, which on short-text embeddings tends to give one giant cluster plus
   singletons.
5. Description: c-TF-IDF terms (Grootendorst 2022) per cluster and the 8 sentences closest to the
   centroid. This is the input for naming the themes (the LLM drafts names, the human approves).
Outputs go to research/results/themes/.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_analysis.preprocessing.lexical import BASE_STOPWORDS  # noqa: E402
from review_analysis.preprocessing.normalize import words  # noqa: E402
from review_analysis.sentiment import SentimentRouter  # noqa: E402

SNAP = ROOT / "data/snapshots/notion_2026-10-13"
OUT = ROOT / "research/results/themes"
EMBEDDER = ("intfloat/multilingual-e5-base", "d128750597")
K_RANGE = range(8, 31)
MIN_WORDS = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sentence_table() -> list[dict[str, Any]]:
    processed = read_jsonl(SNAP / "processed.jsonl")
    rows = []
    for p in processed:
        if p["flags"]["duplicate_of"]:
            continue
        for i, s in enumerate(p["sentences"]):
            if len(words(s)) >= MIN_WORDS:
                rows.append(
                    {
                        "review_id": p["review_id"],
                        "idx": i,
                        "text": s,
                        "lang": p["lang"],
                        "rating": p["rating"],
                    }
                )
    res = SentimentRouter().predict([r["text"] for r in rows], [r["lang"] for r in rows])
    for r, s in zip(rows, res, strict=True):
        r.update(
            label=s.label, p_complaint=round(s.p_complaint, 4), p_negative=round(s.p_negative, 4)
        )
        r["kind"] = (
            "complaint"
            if s.label == "negative" or s.complaint
            else "praise"
            if s.label == "positive"
            else "other"
        )
    return rows


def c_tfidf(texts_by_cluster: dict[int, list[str]], top: int = 10) -> dict[int, list[str]]:
    """Class-based TF-IDF: each cluster's sentences concatenated into one document."""
    ids = sorted(texts_by_cluster)
    stop = sorted(BASE_STOPWORDS | {"notion", "app", "application"})
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words=stop, lowercase=True)
    counts = vec.fit_transform([" ".join(texts_by_cluster[c]) for c in ids]).toarray().astype(float)
    tf = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    avg = counts.sum() / len(ids)
    idf = np.log(1 + avg / np.maximum(counts.sum(0), 1))
    scores = tf * idf
    names = vec.get_feature_names_out()
    return {c: [str(names[j]) for j in np.argsort(-scores[i])[:top]] for i, c in enumerate(ids)}


def cluster(rows: list[dict[str, Any]], emb: np.ndarray, kind: str) -> dict[str, Any]:
    sil = {}
    labels_by_k = {}
    for k in K_RANGE:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(emb)
        sil[k] = float(silhouette_score(emb, lab, metric="cosine"))
        labels_by_k[k] = lab
    best_k = max(sil, key=sil.get)
    labels = labels_by_k[best_k]
    by_c: dict[int, list[int]] = {}
    for i, c in enumerate(labels):
        by_c.setdefault(int(c), []).append(i)
    terms = c_tfidf({c: [rows[i]["text"] for i in idx] for c, idx in by_c.items()})
    clusters = []
    for c, idx in sorted(by_c.items(), key=lambda kv: -len(kv[1])):
        centroid = emb[idx].mean(0)
        centroid /= np.linalg.norm(centroid)
        closest = sorted(idx, key=lambda i: -float(emb[i] @ centroid))[:8]
        clusters.append(
            {
                "cluster": c,
                "n_sentences": len(idx),
                "n_reviews": len({rows[i]["review_id"] for i in idx}),
                "mean_rating": float(np.mean([rows[i]["rating"] for i in idx])),
                "terms": terms[c],
                "examples": [rows[i]["text"] for i in closest],
                "langs": dict(Counter(rows[i]["lang"] for i in idx).most_common(4)),
            }
        )
    for i, c in enumerate(labels):
        rows[i][f"{kind}_cluster"] = int(c)
    return {
        "kind": kind,
        "n_sentences": len(rows),
        "k": best_k,
        "silhouette": sil,
        "clusters": clusters,
    }


def main() -> None:
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    rows = sentence_table()
    print("sentences:", len(rows), dict(Counter(r["kind"] for r in rows)), flush=True)
    model = SentenceTransformer(EMBEDDER[0], revision=EMBEDDER[1], device="cpu")
    for kind in ("complaint", "praise"):
        subset = [r for r in rows if r["kind"] == kind]
        emb = model.encode(
            [f"query: {r['text']}" for r in subset],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = cluster(subset, np.asarray(emb), kind)
        (OUT / f"clusters_{kind}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(
            f"{kind}: {len(subset)} sentences -> k={result['k']} "
            f"(silhouette {result['silhouette'][result['k']]:.3f})",
            flush=True,
        )
        for c in result["clusters"]:
            print(
                f"  [{c['cluster']:2d}] n={c['n_sentences']:3d} ★{c['mean_rating']:.1f} "
                f"{', '.join(c['terms'][:7])} | {c['examples'][0][:80]}"
            )
    with (OUT / "sentences.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
