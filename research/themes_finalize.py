"""Turn the LLM theme labels into the snapshot's theme files (read by analysis.py and the API).

    uv run python research/themes_finalize.py

Writes into data/snapshots/notion_2026-10-13/:
- sentences.jsonl     sid, review_id, idx, kind, text, rating, label, p_complaint, translation
- theme_labels.jsonl  sid, themes (ids from data/themes.json), source model
Labels whose theme id is unknown, or whose kind does not match the sentence, are dropped and
counted.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data/snapshots/notion_2026-10-13"


def main() -> None:
    tax = json.loads((ROOT / "data/themes.json").read_text())
    kind_of = {t["id"]: k for k in ("complaint", "praise") for t in tax[k]}
    processed = {
        json.loads(line)["review_id"]: json.loads(line)
        for line in (SNAP / "processed.jsonl").read_text().splitlines()
    }
    rows = [
        json.loads(line)
        for line in (ROOT / "research/results/themes/sentences.jsonl").read_text().splitlines()
    ]
    labels = {
        json.loads(line)["sid"]: json.loads(line)
        for line in (ROOT / "data/labels/themes_qwen.jsonl").read_text().splitlines()
    }
    dropped: Counter[str] = Counter()
    with (
        (SNAP / "sentences.jsonl").open("w", encoding="utf-8") as fs,
        (SNAP / "theme_labels.jsonl").open("w", encoding="utf-8") as fl,
    ):
        for r in rows:
            sid = f"{r['review_id']}:{r['idx']}"
            p = processed[r["review_id"]]
            fs.write(
                json.dumps(
                    {
                        "sid": sid,
                        "review_id": r["review_id"],
                        "idx": r["idx"],
                        "kind": r["kind"],
                        "text": r["text"],
                        "rating": r["rating"],
                        "label": r["label"],
                        "p_complaint": r["p_complaint"],
                        "translation": p["text_en"] if p["translated"] else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if sid not in labels:
                continue
            themes = []
            for t in labels[sid].get("themes", []):
                if kind_of.get(t) != r["kind"]:
                    dropped["kind mismatch" if t in kind_of else "unknown id"] += 1
                    continue
                if t not in themes:
                    themes.append(t)
            fl.write(
                json.dumps({"sid": sid, "themes": themes, "source": labels[sid].get("model")})
                + "\n"
            )
    print("sentences:", len(rows), "| labelled:", len(labels), "| dropped labels:", dict(dropped))


if __name__ == "__main__":
    main()
