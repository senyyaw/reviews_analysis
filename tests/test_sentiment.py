import json
from pathlib import Path

import numpy as np
import pytest

from review_analysis.sentiment import SentimentRouter, decide

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REAL_MODELS = all(
    (MODELS / d / "model.int8.onnx").exists()
    and (MODELS / d / "model.int8.onnx").stat().st_size > 1_000_000  # not an LFS pointer
    for d in ("sentiment-en", "sentiment-multi")
)
needs_models = pytest.mark.skipif(not REAL_MODELS, reason="int8 models not present (git lfs pull)")


def test_negative_priority_rule():
    assert decide(0.30, 0.0, 0.70) == "negative"  # P(neg) >= 0.30 wins even when not the max
    assert decide(0.29, 0.20, 0.51) == "positive"
    assert decide(0.10, 0.60, 0.30) == "neutral"


class FakeModel:
    def __init__(self, name: str, row: list[float]) -> None:
        self.name, self.row, self.calls = name, row, 0

    def predict_proba(self, texts):  # type: ignore[no-untyped-def]
        self.calls += 1
        return np.array([self.row] * len(texts))


def test_router_sends_each_language_to_its_model():
    router = SentimentRouter.__new__(SentimentRouter)
    router.english = FakeModel("en", [0.9, 0.05, 0.05, 0.8])  # type: ignore[assignment]
    router.multilingual = FakeModel("multi", [0.0, 0.1, 0.9, 0.1])  # type: ignore[assignment]
    res = router.predict(["bad", "muy bien", "meh"], ["en", "es", "und"])
    assert [r.model for r in res] == ["en", "multi", "en"]
    assert [r.label for r in res] == ["negative", "positive", "negative"]
    assert res[0].complaint and not res[1].complaint
    assert router.english.calls == 1  # one batched call per model


@needs_models
def test_real_models_on_clear_cases():
    router = SentimentRouter(MODELS)
    res = router.predict(
        [
            "Love it. Best notes app I have ever used, life changer",
            "Unusable. Crashes every time I open a page, total waste of money",
            "Love it, but I hate that there are limited blocks",
            "Une perte de temps. Le pire logiciel, il est inutilisable.",
        ],
        ["en", "en", "en", "fr"],
    )
    assert [r.label for r in res] == ["positive", "negative", "positive", "negative"]
    assert not res[0].complaint and res[1].complaint and res[2].complaint
    assert res[3].model == "sentiment-multi"


@needs_models
def test_int8_router_matches_fp32_training_predictions():
    """The shipped int8 models must agree with the fp32 checkpoints evaluated in research/."""
    processed = {
        (p := json.loads(line))["review_id"]: p
        for line in (ROOT / "data/snapshots/notion_2026-10-13/processed.jsonl")
        .read_text()
        .splitlines()
    }
    fp32 = {
        "sentiment-en": json.loads(
            (ROOT / "research/results/ft/cardiff-roberta-base_s1/preds.json").read_text()
        )["D1"],
        "sentiment-multi": json.loads(
            (ROOT / "research/results/ft/clapai-xlmr-base_s0/preds.json").read_text()
        )["D1"],
    }
    ids = sorted(fp32["sentiment-en"])[:80]
    router = SentimentRouter(MODELS)
    res = router.predict(
        [processed[r]["text_model"] for r in ids], [processed[r]["lang"] for r in ids]
    )
    agree = [
        int(np.argmax(fp32[r.model][rid][:3]))
        == int(np.argmax([r.p_negative, r.p_neutral, r.p_positive]))
        for rid, r in zip(ids, res, strict=True)
    ]
    assert np.mean(agree) >= 0.95
