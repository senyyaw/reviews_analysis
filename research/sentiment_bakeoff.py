"""Sentiment model comparison (README, Sentiment).

    uv run --extra translate --extra research python research/sentiment_bakeoff.py

Evaluation sets:
  D1-human  100 Notion reviews labelled by the human (star-stratified). This is the headline.
  D1-LLM    the D1 reviews where both LLM annotators agree (natural star mix). Secondary.
  D2        250 reviews from 14 other productivity apps, 50 per star, star labels. Dev set: used
            only to pick ensemble members and to check the decision threshold, never for reporting.
Candidates: VADER, TF-IDF + logistic regression (trained on the other apps' stars), four Hugging
Face models (pinned revisions), an ensemble of the two best on D2, an always-negative baseline,
and the two LLM annotators as a non-deployable reference.

Decision rules: argmax, and the negative-priority rule (negative if P(neg) >= 0.30).
Metrics: macro-F1 with a bootstrap 95% CI, accuracy, MCC, Cohen's kappa, per-class
precision/recall/F1, negative recall/precision, complaint recall (human complaint label) and exact
McNemar tests against the best model. Also CPU latency and model size.
"""

import json
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import emoji
import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_analysis.preprocessing.normalize import (  # noqa: E402
    URL_PLACEHOLDER,
    join_title_body,
    normalize_text,
)

SNAP = ROOT / "data/snapshots/notion_2026-10-13"
OTHERS = ROOT / "data/snapshots/others"
LABELS = ROOT / "data/labels"
OUT = ROOT / "research/results"
CLASSES = ["negative", "neutral", "positive"]  # probability vector order everywhere
NEG_THRESHOLD = 0.30
SEED = 99

HF_MODELS = {
    "cardiffnlp-roberta": ("cardiffnlp/twitter-roberta-base-sentiment-latest", "3216a57f2a"),
    "nlptown-mbert": ("nlptown/bert-base-multilingual-uncased-sentiment", "8f6f4e3a8f"),
    "cardiffnlp-xlmt": ("cardiffnlp/twitter-xlm-roberta-base-sentiment", "f2f1202b1b"),
    "clapai-xlmr": ("clapAI/roberta-base-multilingual-sentiment", "5986405bd6"),
}


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["review_id"]] = row
    return out


def stars_to_label(rating: int) -> str:
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


# --- data ---------------------------------------------------------------------------------------


def load_sets() -> tuple[
    dict[str, dict[str, str]], dict[str, str], list[tuple[str, int]], set[str]
]:
    """Returns (eval sets {name: {review_id: gold}}, texts {id: text}, training data, d2 ids)."""
    processed = read_jsonl(SNAP / "processed.jsonl")
    texts = {rid: p["text_model"] for rid, p in processed.items()}
    human = read_jsonl(LABELS / "human_labels.jsonl")
    qwen = read_jsonl(LABELS / "llm_labels_qwen3.8-27b.jsonl")
    gemma = read_jsonl(LABELS / "llm_labels_gemma4-31b.jsonl")
    spec = json.loads((LABELS / "D1_test_set.json").read_text())

    d1_human = {rid: human[rid]["sentiment"] for rid in spec["human_subset_ids"]}
    d1_llm = {
        rid: qwen[rid]["sentiment"]
        for rid in spec["review_ids"]
        if qwen[rid]["sentiment"] == gemma[rid]["sentiment"]
    }

    others: list[dict[str, Any]] = []
    for pool in sorted(OTHERS.glob("*/pool.jsonl")):
        others += [json.loads(line) for line in pool.read_text().splitlines() if line]
    for r in others:
        texts[r["review_id"]] = normalize_text(join_title_body(r["title"], r["body"]))
    rng = random.Random(SEED)
    by_star: dict[int, list[dict[str, Any]]] = {}
    for r in sorted(others, key=lambda r: r["review_id"]):
        by_star.setdefault(r["rating"], []).append(r)
    d2 = [r for s in range(1, 6) for r in rng.sample(by_star[s], 50)]
    d2_ids = {r["review_id"] for r in d2}
    (LABELS / "D2_dev_set.json").write_text(
        json.dumps(
            {
                "description": "Dev set: 50 reviews per star from 14 other productivity apps "
                "(not Notion). Star labels. Used for ensemble selection and threshold checks only.",
                "seed": SEED,
                "review_ids": sorted(d2_ids),
            },
            indent=1,
        )
    )
    train = [(texts[r["review_id"]], r["rating"]) for r in others if r["review_id"] not in d2_ids]
    sets = {
        "D1-human": d1_human,
        "D1-LLM": d1_llm,
        "D2": {r["review_id"]: stars_to_label(r["rating"]) for r in d2},
    }
    return sets, texts, train, d2_ids


# --- models -------------------------------------------------------------------------------------

Predictor = Callable[[Sequence[str]], np.ndarray]  # texts -> (n, 3) probabilities in CLASSES order


def hf_predictor(repo: str, revision: str) -> tuple[Predictor, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    # transformers 5 loads the checkpoint's saved dtype; clapAI is stored in fp16, which on CPU is
    # ~10x slower and numerically different. Evaluate every model in fp32.
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, revision=revision, dtype=torch.float32
    ).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    unk_emoji = tok.convert_tokens_to_ids(tok.tokenize("🔥")) == [tok.unk_token_id]
    n_params = sum(p.numel() for p in model.parameters())

    def to_three(probs: np.ndarray) -> np.ndarray:
        if len(id2label) == 5:  # nlptown "1 star".."5 stars"
            return np.stack([probs[:, 0] + probs[:, 1], probs[:, 2], probs[:, 3] + probs[:, 4]], 1)
        order = [next(i for i, lab in id2label.items() if lab.startswith(c[:3])) for c in CLASSES]
        return probs[:, order]

    def prep(text: str) -> str:
        text = text.replace(URL_PLACEHOLDER, "http")  # cardiffnlp's own URL convention
        return emoji.demojize(text, delimiters=(" ", " ")) if unk_emoji else text

    def predict(texts: Sequence[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = tok(
                    [prep(t) for t in texts[i : i + 16]],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                out.append(torch.softmax(model(**batch).logits, -1).numpy())
        return to_three(np.concatenate(out))

    info = {
        "repo": repo,
        "revision": revision,
        "params_m": round(n_params / 1e6),
        "emoji_demojized": unk_emoji,
        "labels": id2label,
    }
    return predict, info


def vader_predictor() -> tuple[Predictor, dict[str, Any]]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    def predict(texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            c = sia.polarity_scores(t)["compound"]  # standard thresholds +-0.05, one-hot
            rows.append([1.0, 0, 0] if c <= -0.05 else [0, 0, 1.0] if c >= 0.05 else [0, 1.0, 0])
        return np.array(rows)

    return predict, {"rule": "compound <= -0.05 neg, >= 0.05 pos", "probabilistic": False}


def tfidf_predictor(train: list[tuple[str, int]]) -> tuple[Predictor, dict[str, Any]]:
    labels = [CLASSES.index(stars_to_label(r)) for _, r in train]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
    clf.fit(vec.fit_transform([t for t, _ in train]), labels)

    def predict(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(clf.predict_proba(vec.transform(list(texts))))

    return predict, {
        "train_n": len(train),
        "train_dist": dict(Counter(labels)),
        "vocab": len(vec.vocabulary_),
    }


# --- metrics ------------------------------------------------------------------------------------


def decide(probs: np.ndarray, rule: str) -> list[str]:
    preds = []
    for p in probs:
        if rule == "neg@0.30" and p[0] >= NEG_THRESHOLD:
            preds.append("negative")
        elif rule == "neg@0.30":
            preds.append(CLASSES[1 + int(np.argmax(p[1:]))])
        else:
            preds.append(CLASSES[int(np.argmax(p))])
    return preds


def macro_f1_ci(gold: list[str], pred: list[str], n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    g, p = np.array(gold), np.array(pred)
    vals = [
        f1_score(g[idx], p[idx], labels=CLASSES, average="macro", zero_division=0)
        for idx in (rng.integers(0, len(g), len(g)) for _ in range(n))
    ]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(gold: list[str], pred: list[str], complaint: list[bool] | None) -> dict[str, Any]:
    prec, rec, f1, sup = precision_recall_fscore_support(
        gold, pred, labels=CLASSES, zero_division=0
    )
    lo, hi = macro_f1_ci(gold, pred)
    res: dict[str, Any] = {
        "n": len(gold),
        "macro_f1": f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0),
        "macro_f1_ci": [lo, hi],
        "accuracy": accuracy_score(gold, pred),
        "mcc": matthews_corrcoef(gold, pred),
        "kappa": cohen_kappa_score(gold, pred),
        "per_class": {
            c: {"precision": prec[i], "recall": rec[i], "f1": f1[i], "support": int(sup[i])}
            for i, c in enumerate(CLASSES)
        },
    }
    if complaint is not None:
        flagged = [p == "negative" for p in pred]
        tp = sum(f and c for f, c in zip(flagged, complaint, strict=True))
        res["complaint_recall"] = tp / max(1, sum(complaint))
        res["complaint_precision"] = tp / max(1, sum(flagged))
    return res


def mcnemar(gold: list[str], a: list[str], b: list[str]) -> float:
    a_only = sum(x == g != y for g, x, y in zip(gold, a, b, strict=True))
    b_only = sum(y == g != x for g, x, y in zip(gold, a, b, strict=True))
    return 1.0 if a_only + b_only == 0 else float(binomtest(a_only, a_only + b_only).pvalue)


def latency_ms(predict: Predictor, texts: Sequence[str]) -> float:
    predict(texts[:16])  # warm-up
    start = time.perf_counter()
    predict(texts)
    return 1000 * (time.perf_counter() - start) / len(texts)


# --- main ---------------------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets, texts, train, _ = load_sets()
    human = read_jsonl(LABELS / "human_labels.jsonl")
    all_ids = sorted({rid for s in sets.values() for rid in s})
    print({k: (len(v), dict(Counter(v.values()))) for k, v in sets.items()}, flush=True)

    probs: dict[str, dict[str, np.ndarray]] = {}
    info: dict[str, dict[str, Any]] = {}
    builders: dict[str, Callable[[], tuple[Predictor, dict[str, Any]]]] = {
        "vader": vader_predictor,
        "tfidf-logreg": lambda: tfidf_predictor(train),
        **{
            name: (lambda r=repo, v=rev: hf_predictor(r, v))
            for name, (repo, rev) in HF_MODELS.items()
        },
    }
    timing_texts = [texts[r] for r in sorted(sets["D1-human"])][:64]
    for name, build in builders.items():
        print(f"running {name} ...", flush=True)
        predict, meta = build()
        p = predict([texts[r] for r in all_ids])
        probs[name] = dict(zip(all_ids, p, strict=True))
        meta["cpu_ms_per_review"] = round(latency_ms(predict, timing_texts), 1)
        info[name] = meta

    # Ensemble: average the two best probabilistic models, chosen on D2 (dev), never on D1.
    d2_gold = sets["D2"]
    d2_scores = {
        name: f1_score(
            [d2_gold[r] for r in d2_gold],
            decide(np.array([probs[name][r] for r in d2_gold]), "argmax"),
            labels=CLASSES,
            average="macro",
        )
        for name in probs
        if name != "vader"
    }
    top2 = sorted(d2_scores, key=d2_scores.get, reverse=True)[:2]
    probs["ensemble"] = {r: (probs[top2[0]][r] + probs[top2[1]][r]) / 2 for r in all_ids}
    info["ensemble"] = {
        "members": top2,
        "chosen_on": "D2 macro-F1 (argmax)",
        "d2_scores": d2_scores,
    }

    one_hot = {c: np.eye(3)[i] for i, c in enumerate(CLASSES)}
    probs["always-negative"] = {r: one_hot["negative"] for r in all_ids}
    for llm in ("qwen3.8-27b", "gemma4-31b"):
        lab = read_jsonl(LABELS / f"llm_labels_{llm}.jsonl")
        probs[f"llm:{llm}"] = {r: one_hot[lab[r]["sentiment"]] for r in all_ids if r in lab}

    results: dict[str, Any] = {
        "sets": {k: len(v) for k, v in sets.items()},
        "models": info,
        "threshold_check": {},
        "eval": {},
    }
    for set_name, gold_map in sets.items():
        ids = sorted(gold_map)
        gold = [gold_map[r] for r in ids]
        complaint = [bool(human[r]["complaint"]) for r in ids] if set_name == "D1-human" else None
        for name, pmap in probs.items():
            if not all(r in pmap for r in ids):
                continue
            mat = np.array([pmap[r] for r in ids])
            for rule in ("argmax", "neg@0.30"):
                pred = decide(mat, rule)
                results["eval"].setdefault(set_name, {}).setdefault(name, {})[rule] = {
                    **evaluate(gold, pred, complaint),
                    "_pred": pred,
                }

    # Threshold safety check on D2 (negative precision must stay >= ~0.75).
    for name, by_rule in results["eval"]["D2"].items():
        r = by_rule["neg@0.30"]["per_class"]["negative"]
        results["threshold_check"][name] = {
            "neg_precision@0.30": r["precision"],
            "neg_recall@0.30": r["recall"],
        }

    # McNemar: best deployable model (D1-human, argmax macro-F1) vs every other model.
    for set_name in ("D1-human", "D1-LLM"):
        ev = results["eval"][set_name]
        deployable = [m for m in ev if not m.startswith("llm:") and m != "always-negative"]
        best = max(deployable, key=lambda m: ev[m]["argmax"]["macro_f1"])
        gold = [sets[set_name][r] for r in sorted(sets[set_name])]
        for m in ev:
            ev[m]["argmax"]["mcnemar_p_vs_best"] = mcnemar(
                gold, ev[best]["argmax"]["_pred"], ev[m]["argmax"]["_pred"]
            )
        results.setdefault("best_deployable", {})[set_name] = best

    for ev in results["eval"].values():
        for by_rule in ev.values():
            for r in by_rule.values():
                r.pop("_pred")
    (OUT / "sentiment_bakeoff.json").write_text(json.dumps(results, indent=1, default=float))
    print_tables(results)


def print_tables(results: dict[str, Any]) -> None:
    for set_name, ev in results["eval"].items():
        print(f"\n### {set_name} (n={results['sets'][set_name]})")
        print(
            "| model | macro-F1 [95% CI] | acc | MCC | neg R / P (argmax) | neg R / P (@0.30) |"
            " complaint R (@0.30) | McNemar p vs best |"
        )
        print("|---|---|---|---|---|---|---|---|")
        ranked = sorted(ev, key=lambda m: -ev[m]["argmax"]["macro_f1"])
        for m in ranked:
            a, t = ev[m]["argmax"], ev[m]["neg@0.30"]
            na, nt = a["per_class"]["negative"], t["per_class"]["negative"]
            cr = f"{t['complaint_recall']:.2f}" if "complaint_recall" in t else "-"
            mp = a.get("mcnemar_p_vs_best")
            print(
                f"| {m} | {a['macro_f1']:.3f} "
                f"[{a['macro_f1_ci'][0]:.2f}, {a['macro_f1_ci'][1]:.2f}]"
                f" | {a['accuracy']:.2f} | {a['mcc']:.2f}"
                f" | {na['recall']:.2f} / {na['precision']:.2f}"
                f" | {nt['recall']:.2f} / {nt['precision']:.2f} | {cr} |"
                f" {'' if mp is None else f'{mp:.3f}'} |"
            )
    print(
        "\nlatency (CPU, ms/review):",
        {m: i.get("cpu_ms_per_review") for m, i in results["models"].items()},
    )
    print("ensemble:", results["models"]["ensemble"]["members"])
    print(
        "threshold check on D2:",
        {m: round(v["neg_precision@0.30"], 2) for m, v in results["threshold_check"].items()},
    )
    print("best deployable:", results["best_deployable"])


if __name__ == "__main__":
    main()
