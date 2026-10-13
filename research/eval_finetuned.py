"""Evaluate the fine-tuned models with exactly the same metrics as the off-the-shelf comparison.

    uv run --extra research python research/eval_finetuned.py --runs research/results/ft

Expects one directory per run (<model>_s<seed>/ with preds.json and meta.json), copied from the
server's out/ft/. For every model it reports:
- each seed, and the mean ± sd over seeds;
- a seed ensemble (probabilities averaged over the 3 seeds);
- cross-model ensembles of the top-k seed ensembles. Members are ranked on the VALIDATION split
  (other apps), never on Notion data.

Sets: D1-human (headline), D1-LLM (LLM-consensus labels), D2-LLM (other apps, LLM-consensus labels;
replaces star labels for the threshold check). Complaint: F1, recall and precision of the complaint
head (p >= 0.5) against the human complaint labels (D1-human) and the LLM labels (D1-LLM).
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import sentiment_bakeoff as sb  # noqa: E402

LABELS = ROOT / "data/labels"


def consensus(qwen: Path, gemma: Path) -> dict[str, dict[str, Any]]:
    q, g = sb.read_jsonl(qwen), sb.read_jsonl(gemma)
    return {
        r: {"sentiment": q[r]["sentiment"], "complaint": bool(q[r]["complaint"])}
        for r in q
        if r in g
        and q[r]["sentiment"] == g[r]["sentiment"]
        and q[r]["complaint"] == g[r]["complaint"]
    }


def complaint_scores(gold: list[bool], prob: list[float]) -> dict[str, float]:
    pred = [p >= 0.5 for p in prob]
    tp = sum(a and b for a, b in zip(pred, gold, strict=True))
    return {
        "complaint_f1": f1_score(gold, pred, zero_division=0),
        "complaint_recall": tp / max(1, sum(gold)),
        "complaint_precision": tp / max(1, sum(pred)),
    }


def score(
    probs: dict[str, list[float]], gold: dict[str, dict[str, Any]], rule: str
) -> dict[str, Any]:
    ids = sorted(r for r in gold if r in probs)
    mat = np.array([probs[r] for r in ids])
    pred = sb.decide(mat[:, :3], rule)
    res = sb.evaluate([gold[r]["sentiment"] for r in ids], pred, None)
    res |= complaint_scores([gold[r]["complaint"] for r in ids], list(mat[:, 3]))
    res["_pred"] = pred
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "research/results/ft"))
    args = ap.parse_args()

    human = sb.read_jsonl(LABELS / "human_labels.jsonl")
    spec = json.loads((LABELS / "D1_test_set.json").read_text())
    sets = {
        "D1-human": {
            r: {"sentiment": human[r]["sentiment"], "complaint": bool(human[r]["complaint"])}
            for r in spec["human_subset_ids"]
        },
        "D1-LLM": consensus(
            LABELS / "llm_labels_qwen3.8-27b.jsonl", LABELS / "llm_labels_gemma4-31b.jsonl"
        ),
    }
    d2_ids = set(json.loads((LABELS / "D2_dev_set.json").read_text())["review_ids"])
    others = consensus(
        LABELS / "others_labels_qwen3.8-27b.jsonl", LABELS / "others_labels_gemma4-31b.jsonl"
    )
    sets["D2-LLM"] = {r: v for r, v in others.items() if r in d2_ids}

    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in sorted(Path(args.runs).glob("*_s[0-9]")):
        if (d / "preds.json").exists():
            runs[d.name.rsplit("_s", 1)[0]].append(
                {
                    "seed": d.name.rsplit("_s", 1)[1],
                    "preds": json.loads((d / "preds.json").read_text()),
                    "meta": json.loads((d / "meta.json").read_text()),
                }
            )

    def avg(pred_list: list[dict[str, list[float]]]) -> dict[str, list[float]]:
        return {r: list(np.mean([p[r] for p in pred_list], axis=0)) for r in pred_list[0]}

    systems: dict[str, dict[str, Any]] = {}  # name -> {"D1": probs, "D2": probs, "val": array}
    report: dict[str, Any] = {"per_seed": {}, "systems": {}}
    for model, rs in runs.items():
        seed_scores = defaultdict(list)
        for run in rs:
            for set_name, gold in sets.items():
                key = "D2" if set_name.startswith("D2") else "D1"
                s = score(run["preds"][key], gold, "argmax")
                seed_scores[set_name].append(s["macro_f1"])
        report["per_seed"][model] = {
            k: {
                "mean": statistics.fmean(v),
                "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
                "values": v,
            }
            for k, v in seed_scores.items()
        }
        val_gold = [g[0] for g in rs[0]["preds"]["val_gold"]]
        val = np.mean([np.array(r["preds"]["val"]) for r in rs], axis=0)
        systems[f"{model} (seed-ens)"] = {
            "D1": avg([r["preds"]["D1"] for r in rs]),
            "D2": avg([r["preds"]["D2"] for r in rs]),
            "val_f1": f1_score(val_gold, val[:, :3].argmax(1), average="macro"),
            "val": val,
            "params_m": rs[0]["meta"]["params_m"],
        }

    ranked = sorted(systems, key=lambda s: -systems[s]["val_f1"])  # ranked on VALIDATION only
    for k in (2, 3):
        members = ranked[:k]
        systems[f"ensemble top-{k} by val"] = {
            "D1": avg([systems[m]["D1"] for m in members]),
            "D2": avg([systems[m]["D2"] for m in members]),
            "val_f1": f1_score(
                val_gold,
                np.mean([systems[m]["val"] for m in members], 0)[:, :3].argmax(1),
                average="macro",
            ),
            "members": members,
        }

    for name, sysd in systems.items():
        report["systems"][name] = {
            "val_macro_f1": sysd["val_f1"],
            "members": sysd.get("members"),
            "params_m": sysd.get("params_m"),
        }
        for set_name, gold in sets.items():
            key = "D2" if set_name.startswith("D2") else "D1"
            for rule in ("argmax", "neg@0.30"):
                report["systems"][name].setdefault(set_name, {})[rule] = score(
                    sysd[key], gold, rule
                )

    # McNemar vs the off-the-shelf winner on the two D1 sets.
    base = json.loads((ROOT / "research/results/sentiment_bakeoff.json").read_text())
    print(
        "off-the-shelf cardiffnlp D1-human macro-F1:",
        round(base["eval"]["D1-human"]["cardiffnlp-roberta"]["argmax"]["macro_f1"], 3),
    )
    out = {k: v for k, v in report.items()}
    for sysd in out["systems"].values():
        for set_name in sets:
            for r in sysd[set_name].values():
                r.pop("_pred", None)
    (ROOT / "research/results/finetune_eval.json").write_text(
        json.dumps(out, indent=1, default=float)
    )

    print(
        "\n| system | val F1 | D1-human macro-F1 [CI] | acc | MCC | neg R/P | complaint F1 (human)"
        " | D1-LLM macro-F1 | D2-LLM neg P@0.30 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for name in sorted(
        report["systems"], key=lambda n: -report["systems"][n]["D1-human"]["argmax"]["macro_f1"]
    ):
        s = report["systems"][name]
        h, l2, d2 = s["D1-human"]["argmax"], s["D1-LLM"]["argmax"], s["D2-LLM"]["neg@0.30"]
        n = h["per_class"]["negative"]
        print(
            f"| {name} | {s['val_macro_f1']:.3f} | {h['macro_f1']:.3f} [{h['macro_f1_ci'][0]:.2f},"
            f" {h['macro_f1_ci'][1]:.2f}] | {h['accuracy']:.2f} | {h['mcc']:.2f} |"
            f" {n['recall']:.2f}/{n['precision']:.2f} | {h['complaint_f1']:.2f} |"
            f" {l2['macro_f1']:.3f} | {d2['per_class']['negative']['precision']:.2f} |"
        )
    print("\nper-seed D1-human macro-F1 (mean ± sd):")
    for m, v in report["per_seed"].items():
        values = ", ".join(f"{x:.3f}" for x in v["D1-human"]["values"])
        print(f"  {m}: {v['D1-human']['mean']:.3f} ± {v['D1-human']['sd']:.3f}  [{values}]")


if __name__ == "__main__":
    main()
