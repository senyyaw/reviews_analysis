"""Charts for the methods section: model comparison (forest plot) and confusion matrices.

    uv run --extra report python research/model_eval_charts.py

Following the evaluation figures in Shanto et al. (arXiv 2605.22003), with the uncertainty those
figures lack: every macro-F1 has its bootstrap 95% CI, and n is stated. The data comes from the
stored results (sentiment_bakeoff.json, finetune_eval.json, deployed_int8_eval.json). The confusion
matrices recompute the shipped int8 router's predictions on the 100 human-labelled reviews.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from review_analysis import viz  # noqa: E402
from review_analysis.sentiment import SentimentRouter  # noqa: E402

RES = ROOT / "research/results"
OUT = ROOT / "reports"
CLASSES = ["negative", "neutral", "positive"]


def comparison_rows() -> list[dict[str, object]]:
    base = json.loads((RES / "sentiment_bakeoff.json").read_text())["eval"]["D1-human"]
    ft = json.loads((RES / "finetune_eval.json").read_text())["systems"]
    dep = json.loads((RES / "deployed_int8_eval.json").read_text())["D1-human argmax"]
    rows: list[dict[str, object]] = []

    def add(name: str, r: dict[str, object], group: str) -> None:
        rows.append({"name": name, "f1": r["macro_f1"], "ci": r["macro_f1_ci"], "group": group})

    add("always negative (baseline)", base["always-negative"]["argmax"], "baseline")
    add("VADER (lexicon)", base["vader"]["argmax"], "off-the-shelf")
    add("TF-IDF + logistic regression", base["tfidf-logreg"]["argmax"], "off-the-shelf")
    for key, label in (
        ("nlptown-mbert", "nlptown mBERT"),
        ("clapai-xlmr", "clapAI XLM-R"),
        ("cardiffnlp-xlmt", "cardiffnlp XLM-T"),
        ("cardiffnlp-roberta", "cardiffnlp RoBERTa"),
    ):
        add(label, base[key]["argmax"], "off-the-shelf")
    for key, label in (
        ("mdeberta-v3-base (seed-ens)", "mDeBERTa-v3-base"),
        ("xlmr-large (seed-ens)", "XLM-R-large"),
        ("clapai-xlmr-base (seed-ens)", "clapAI XLM-R"),
        ("deberta-v3-base (seed-ens)", "DeBERTa-v3-base"),
        ("roberta-large (seed-ens)", "RoBERTa-large"),
        ("ensemble top-2 by val", "ensemble (top-2 by val)"),
        ("deberta-v3-large (seed-ens)", "DeBERTa-v3-large"),
        ("cardiff-roberta-base (seed-ens)", "cardiffnlp RoBERTa"),
    ):
        add(f"{label}, fine-tuned", ft[key]["D1-human"]["argmax"], "fine-tuned")
    add("SHIPPED: router, int8 ONNX", dep, "shipped")
    add("LLM annotator (Qwen3.8-27B)", base["llm:qwen3.8-27b"]["argmax"], "reference")
    return rows


def forest(rows: list[dict[str, object]], path: Path) -> None:
    viz._style()
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    colors = {
        "baseline": viz.MUTED,
        "off-the-shelf": viz.AXIS,
        "fine-tuned": viz.BLUE,
        "shipped": viz.ORANGE,
        "reference": viz.INK2,
    }
    for y, r in enumerate(rows):
        lo, hi = r["ci"]  # type: ignore[misc]
        ax.plot(
            [lo, hi], [y, y], color=colors[str(r["group"])], linewidth=2, solid_capstyle="round"
        )
        ax.scatter(
            [r["f1"]],
            [y],
            s=60,
            color=colors[str(r["group"])],
            edgecolor=viz.SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            float(hi) + 0.01, y, f"{float(r['f1']):.2f}", va="center", fontsize=8.5, color=viz.INK2
        )  # type: ignore[arg-type]
    ax.set_yticks(range(len(rows)), [str(r["name"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel("macro-F1 on the 100 human-labelled Notion reviews (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    shipped = next(r for r in rows if r["group"] == "shipped")
    start = next(r for r in rows if r["name"] == "cardiffnlp RoBERTa")
    viz._save(
        fig,
        path,
        f"Fine-tuning lifted macro-F1 from {float(start['f1']):.2f} to {float(shipped['f1']):.2f}; "
        "bigger models and ensembles added nothing",
        "Every candidate evaluated on the same human test set · fine-tuned = seed ensemble over 3 "
        "seeds, distilled from LLM labels on 14 other apps",
        legend=[
            viz._key(viz.AXIS, "off-the-shelf", "o"),
            viz._key(viz.BLUE, "fine-tuned", "o"),
            viz._key(viz.ORANGE, "shipped", "o"),
            viz._key(viz.INK2, "LLM reference", "o"),
        ],
    )


def confusion(path: Path) -> dict[str, object]:
    human = {
        json.loads(line)["review_id"]: json.loads(line)
        for line in (ROOT / "data/labels/human_labels.jsonl").read_text().splitlines()
    }
    proc = {
        json.loads(line)["review_id"]: json.loads(line)
        for line in (ROOT / "data/snapshots/notion_2026-10-13/processed.jsonl")
        .read_text()
        .splitlines()
    }
    ids = json.loads((ROOT / "data/labels/D1_test_set.json").read_text())["human_subset_ids"]
    res = SentimentRouter().predict(
        [proc[r]["text_model"] for r in ids], [proc[r]["lang"] for r in ids]
    )
    sent = np.zeros((3, 3), dtype=int)
    comp = np.zeros((2, 2), dtype=int)
    for r, p in zip(ids, res, strict=True):
        sent[CLASSES.index(human[r]["sentiment"]), CLASSES.index(p.label)] += 1
        comp[int(bool(human[r]["complaint"])), int(p.complaint)] += 1

    viz._style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    for ax, mat, labels, title in (
        (ax1, sent, CLASSES, "Sentiment"),
        (ax2, comp, ["no complaint", "complaint"], "Complaint"),
    ):
        norm = mat / np.maximum(mat.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=viz.BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if norm[i, j] > 0.55 else viz.INK,
                )
        ax.set_xticks(range(len(labels)), [f"pred: {lab}" for lab in labels], fontsize=8.5)
        ax.set_yticks(range(len(labels)), [f"human: {lab}" for lab in labels], fontsize=8.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=10, color=viz.INK2)
    acc = np.trace(sent) / sent.sum()
    c_recall = comp[1, 1] / max(1, comp[1].sum())
    viz._save(
        fig,
        path,
        f"Shipped model: {acc:.0%} sentiment accuracy, catches {c_recall:.0%} of complaints",
        "Counts on the 100 human-labelled reviews (star-stratified) · shading = share of the "
        "row · most errors are the rare neutral class",
    )
    return {"sentiment": sent.tolist(), "complaint": comp.tolist(), "labels": CLASSES}


def main() -> None:
    rows = comparison_rows()
    forest(rows, OUT / "figures" / "model_comparison.png")
    cm = confusion(OUT / "figures" / "confusion_matrix.png")
    (OUT / "data" / "model_eval.json").write_text(
        json.dumps({"comparison": rows, "confusion": cm}, indent=1)
    )
    print("wrote model_comparison.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
