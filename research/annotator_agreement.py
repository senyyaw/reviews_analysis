"""How far can we trust the LLM labels? Agreement between the human and the LLM annotators on D1.

    uv run python research/annotator_agreement.py

It reports, for sentiment and for complaint:
- Cohen's kappa (with a bootstrap 95% CI) and raw agreement, human vs each LLM, on the human subset;
- the same agreement reweighted by star stratum: the human subset over-samples 2-4 stars, so the
  weights give the estimate for the natural D1 distribution;
- LLM vs LLM kappa on all 300 reviews, plus their disagreements (candidates for adjudication);
- agreement on reviews the human marked "unsure" vs the rest.
Decision rule (DESIGN.md §5.3): the LLM labels are used as a secondary test set only if
human-LLM sentiment kappa >= 0.8.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/labels"
SNAPSHOT = ROOT / "data/snapshots/notion_2026-10-13"
CLASSES = ["positive", "neutral", "negative"]


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["review_id"]] = row  # latest entry wins (relabelling in the tool)
    return out


def kappa_ci(
    a: list[Any], b: list[Any], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    k = cohen_kappa_score(a, b)
    rng = random.Random(seed)
    idx = range(len(a))
    boots = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        aa, bb = [a[i] for i in s], [b[i] for i in s]
        if len(set(aa) | set(bb)) > 1:
            boots.append(cohen_kappa_score(aa, bb))
    boots.sort()
    return k, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def main() -> None:
    spec = json.loads((LABELS / "D1_test_set.json").read_text())
    stars = {
        json.loads(line)["review_id"]: json.loads(line)["rating"]
        for line in (SNAPSHOT / "pool.jsonl").read_text().splitlines()
    }
    human = (
        read_jsonl(LABELS / "human_labels.jsonl")
        if (LABELS / "human_labels.jsonl").exists()
        else {}
    )
    llms = {
        p.stem.removeprefix("llm_labels_"): read_jsonl(p)
        for p in sorted(LABELS.glob("llm_labels_*.jsonl"))
    }

    d1_star = Counter(stars[r] for r in spec["review_ids"])
    sub_star = Counter(stars[r] for r in spec["human_subset_ids"])
    weight = {
        s: (d1_star[s] / len(spec["review_ids"])) / (sub_star[s] / len(spec["human_subset_ids"]))
        for s in sub_star
    }

    print(f"human labels: {len(human)}/100 | LLM annotators: {', '.join(llms) or 'none yet'}\n")
    for name, llm in llms.items():
        ids = [r for r in human if r in llm and llm[r]["sentiment"]]
        if len(ids) < 10:
            continue
        print(f"=== human vs {name}  (n={len(ids)})")
        for field in ("sentiment", "complaint"):
            h = [human[r][field] for r in ids]
            m = [llm[r][field] for r in ids]
            k, lo, hi = kappa_ci(h, m)
            agree = [x == y for x, y in zip(h, m, strict=True)]
            w = [weight[stars[r]] for r in ids]
            weighted = sum(wi for wi, ok in zip(w, agree, strict=True) if ok) / sum(w)
            print(
                f"  {field:9s} kappa {k:.3f} [{lo:.3f}, {hi:.3f}] "
                f"| agreement {sum(agree) / len(agree):.1%}"
                f" (star-reweighted {weighted:.1%})"
            )
        h = [human[r]["sentiment"] for r in ids]
        m = [llm[r]["sentiment"] for r in ids]
        print("  confusion (rows = human, cols = LLM):", CLASSES)
        for label, row in zip(CLASSES, confusion_matrix(h, m, labels=CLASSES), strict=True):
            print(f"    {label:9s} {[int(x) for x in row]}")
        unsure = [r for r in ids if human[r].get("unsure")]
        if unsure:
            ok = sum(human[r]["sentiment"] == llm[r]["sentiment"] for r in unsure)
            print(f"  on human-'unsure' reviews: {ok}/{len(unsure)} agree")
        print()

    names = list(llms)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = [
                r
                for r in spec["review_ids"]
                if llms[a].get(r, {}).get("sentiment") and llms[b].get(r, {}).get("sentiment")
            ]
            for field in ("sentiment", "complaint"):
                k, lo, hi = kappa_ci(
                    [llms[a][r][field] for r in ids], [llms[b][r][field] for r in ids]
                )
                print(f"=== {a} vs {b} {field}: kappa {k:.3f} [{lo:.3f}, {hi:.3f}] (n={len(ids)})")
            disagree = [r for r in ids if llms[a][r]["sentiment"] != llms[b][r]["sentiment"]]
            print(
                f"    sentiment disagreements: {len(disagree)} "
                "-> data/labels/llm_disagreements.json"
            )
            (LABELS / "llm_disagreements.json").write_text(json.dumps(disagree, indent=1))


if __name__ == "__main__":
    main()
